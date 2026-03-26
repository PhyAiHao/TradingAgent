"""
heartbeat.py
============
Autonomous trading loop — single coordinator for the full cycle.

Imports builders exclusively from model_builder.py (variable-N aware).
Does NOT import any builder functions from trading_model.py.

Each cycle:
  1.  Load saved Dirichlet params from disk       (Layer 2 restore)
  2.  Fetch news + macro corpus                   (web search)
  3.  LLM reflection → variable-N llm_params      (Layer 3)
  4.  Fetch Alpaca bars                            (price data)
  5.  Build L1 MDPs with N_sent / N_vol from LLM  (model_builder)
  6.  Build L2 MDP — adapts A tensor dimensions   (model_builder)
  7.  Inject price observations (Ng-aware)         (model_builder)
  8.  Restore Dirichlet params, resizing if N changed
  9.  spm_MDP_VB_X → beliefs + policy             (solver)
  10. Save updated Dirichlet params               (Layer 2 write)
  11. Confidence gate → Alpaca execution           (trade)
  12. Log trade                                    (Layer 1 write)
  13. Update P&L on previous open trades           (Layer 1 update)
  14. Telegram report (N-aware formatting)         (notification)
"""

import os
import sys
import time
import requests
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

load_dotenv()

# ── Pure data functions from trading_model (no dimension assumptions) ─────
from active_inference_engine.trading_model import (
    fetch_l1_observations,
    call_llm_for_l1,
    execute_signal,
    SYMBOL, QTY, MIN_CONFIDENCE,
)

# ── All MDP construction from memory/model_builder ────────────────────────
from memory.model_builder import (
    build_l1_sentiment_mdp,
    build_l1_volatility_mdp,
    build_l2_mdp,
    inject_price_observations,
    beliefs_to_signal,
)

from active_inference_engine.solver import spm_MDP_VB_X
from tools.alpaca_tools import get_historical_bars, get_position

from memory.memory import (
    log_trade,
    update_trade_outcome,
    load_recent_trades,
    save_params,
    llm_reflect,
)

# ── All config from root config.py ────────────────────────────────────────
from config import (
    HEARTBEAT_INTERVAL, MARKET_HOURS_ONLY, SYMBOL as _DEFAULT_SYMBOL,
    T, N_REGIME, QTY as _DEFAULT_QTY, MIN_CONFIDENCE as _DEFAULT_MIN_CONF,
    LLM_MODEL, TIMEFRAME, BAR_LIMIT,
    TELEGRAM_BOT_TOKEN as _BOT_TOKEN,
)

# ===========================================================================
# Runtime overrides (env vars shadow config.py defaults where needed)
# ===========================================================================

HEARTBEAT_SYMBOL   = os.getenv("HEARTBEAT_SYMBOL",   _DEFAULT_SYMBOL)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", _BOT_TOKEN or "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")


# ===========================================================================
# Market hours
# ===========================================================================

def _is_market_open() -> bool:
    if not MARKET_HOURS_ONLY:
        return True
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=13, minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=20, minute=0,  second=0, microsecond=0)
    return open_ <= now <= close_


# ===========================================================================
# Telegram
# ===========================================================================

def _send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=10,
        )
    except Exception as e:
        print(f"[telegram] {e}")


def _format_report(symbol: str, signal: dict, llm_params: dict,
                   result: dict, reflected: bool) -> str:
    """
    Format Telegram report — handles variable N_sent and N_vol.
    Uses state_names from llm_params so the report always matches
    what the LLM actually proposed this cycle.
    """
    regime_names = ["risk-on bull", "risk-off bear", "uncertain", "trending"]
    regime_belief = signal["regime_belief"]
    top_regime    = regime_names[int(np.argmax(regime_belief))]

    action = signal["action"].upper()
    conf   = signal["confidence"]
    order  = result.get("order")

    if result["executed"] and order:
        gate = f"✅ EXECUTED — {order['id']}"
    elif conf < MIN_CONFIDENCE:
        gate = f"⏸ skipped (conf {conf:.2f} < {MIN_CONFIDENCE})"
    else:
        gate = "⏸ HOLD"

    mem_tag = "🧠 memory-reflected" if reflected else "📋 fresh priors"

    # Sentiment beliefs — variable length, use state_names
    sent_names  = signal.get("sent_names", [f"s{i}" for i in range(len(signal["sent_belief"]))])
    sent_str    = "  ".join(
        f"{n}={v:.2f}" for n, v in zip(sent_names, signal["sent_belief"])
    )

    # Volatility beliefs — variable length
    vol_names   = signal.get("vol_names", [f"v{i}" for i in range(len(signal["vol_belief"]))])
    vol_str     = "  ".join(
        f"{n}={v:.2f}" for n, v in zip(vol_names, signal["vol_belief"])
    )

    return (
        f"🤖 Active Inference — {symbol}  [{mem_tag}]\n"
        f"{'─' * 36}\n"
        f"Signal    : {action}  ({gate})\n"
        f"Confidence: {conf:.2f}\n"
        f"Regime    : {top_regime}  {regime_belief.round(2).tolist()}\n"
        f"Sentiment : {sent_str}\n"
        f"Volatility: {vol_str}\n"
        f"States    : sent={llm_params.get('n_sent',3)}  "
        f"vol={llm_params.get('n_vol',3)}\n"
        f"LLM note  : {llm_params['reasoning']}\n"
        f"Time      : {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


# ===========================================================================
# P&L updater
# ===========================================================================

def _update_open_pnl(symbol: str) -> None:
    recent  = load_recent_trades(20)
    pending = [t for t in recent
               if t.get("executed") and t.get("outcome") is None
               and t.get("order_id")]
    if not pending:
        return
    pos = get_position(symbol)
    if "error" in pos:
        return
    update_trade_outcome(
        order_id   = pending[-1]["order_id"],
        exit_price = float(pos.get("current_price", 0)),
        pnl        = float(pos.get("unrealized_pl", 0)),
    )


# ===========================================================================
# Dirichlet param store — load and resize for variable N
# ===========================================================================

def _load_and_resize_params(mdp_l2, mdp_sent, mdp_vol, llm_params: dict):
    """
    Load saved params and apply:
      - L1 params to mdp_sent / mdp_vol (resized if N changed)
      - L2 params to mdp_l2 (skipped if shapes changed)
    """
    from memory.memory import load_params_raw
    from memory.model_builder import resize_params

    payload = load_params_raw()
    if payload is None:
        print("[heartbeat] No saved params — using fresh priors")
        return mdp_sent, mdp_vol, mdp_l2

    print(f"[heartbeat] Restoring params saved at {payload.get('saved_at','?')}")
    n_sent = llm_params.get("n_sent", 3)
    n_vol  = llm_params.get("n_vol",  3)

    # Resize L1 params to current N (handles N change between cycles)
    resized = resize_params(payload, n_sent, n_vol)

    # ── L1 sentiment ──────────────────────────────────────────────────────
    if resized and resized.get("l1_sent_a") is not None:
        mdp_sent.a   = [resized["l1_sent_a"]]
        mdp_sent.a_0 = [resized["l1_sent_a_0"]]
        print(f"[heartbeat] L1 sentiment a restored {resized['l1_sent_a'].shape}")
    if resized and resized.get("l1_sent_d") is not None:
        mdp_sent.d   = [resized["l1_sent_d"]]
        mdp_sent.d_0 = [resized["l1_sent_d_0"]]

    # ── L1 volatility ─────────────────────────────────────────────────────
    if resized and resized.get("l1_vol_a") is not None:
        mdp_vol.a   = [resized["l1_vol_a"]]
        mdp_vol.a_0 = [resized["l1_vol_a_0"]]
        print(f"[heartbeat] L1 volatility a restored {resized['l1_vol_a'].shape}")
    if resized and resized.get("l1_vol_d") is not None:
        mdp_vol.d   = [resized["l1_vol_d"]]
        mdp_vol.d_0 = [resized["l1_vol_d_0"]]

    # ── L2 params — only if shapes still match ────────────────────────────
    if payload.get("l2_a") is not None:
        saved_shapes   = [a.shape for a in payload["l2_a"]]
        current_shapes = [a.shape for a in mdp_l2.a]
        if saved_shapes == current_shapes:
            mdp_l2.a   = [p.copy() for p in payload["l2_a"]]
            mdp_l2.a_0 = [p.copy() for p in payload["l2_a_0"]] \
                          if payload.get("l2_a_0") else mdp_l2.a_0
            print("[heartbeat] L2 a restored")
        else:
            print(f"[heartbeat] L2 a shape changed — skipping restore")

    if payload.get("l2_d") is not None:
        if all(len(s) == len(c) for s, c in zip(payload["l2_d"], mdp_l2.d)):
            mdp_l2.d   = [p.copy() for p in payload["l2_d"]]
            mdp_l2.d_0 = [p.copy() for p in payload["l2_d_0"]] \
                          if payload.get("l2_d_0") else mdp_l2.d_0

    if payload.get("l2_e") is not None and mdp_l2.e is not None:
        if payload["l2_e"].shape == mdp_l2.e.shape:
            mdp_l2.e   = payload["l2_e"].copy()
            mdp_l2.e_0 = payload["l2_e_0"].copy() \
                          if payload.get("l2_e_0") else mdp_l2.e_0

    return mdp_sent, mdp_vol, mdp_l2


# ===========================================================================
# Core cycle
# ===========================================================================

def run_cycle(symbol: str) -> None:
    print(f"\n[heartbeat] {'─' * 44}")
    print(f"[heartbeat] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {symbol}")

    # Step 13: update P&L on previous open trades
    _update_open_pnl(symbol)

    # Step 2: fetch news + macro corpus
    print("[cycle] 1/6  Fetching observations (web search)...")
    corpus = fetch_l1_observations(symbol)

    # Step 3: LLM reflection — returns variable N_sent, N_vol
    print("[cycle] 2/6  LLM reflection on trade history...")
    llm_params = llm_reflect(corpus=corpus, symbol=symbol)
    reflected  = llm_params is not None

    if not reflected:
        # No history yet — standard L1 call, fixed N=3
        print("[cycle]      No history — standard L1 LLM call...")
        llm_params = call_llm_for_l1(corpus, symbol)
        # Ensure n_sent / n_vol keys present
        llm_params.setdefault("n_sent", 3)
        llm_params.setdefault("n_vol",  3)

    n_sent = llm_params["n_sent"]
    n_vol  = llm_params["n_vol"]
    print(f"[cycle]      n_sent={n_sent}  n_vol={n_vol}  "
          f"reasoning: {llm_params['reasoning']}")

    pref_profit = float(llm_params.get("pref_profit", 2.0))
    pref_loss   = float(llm_params.get("pref_loss",  -3.0))

    # Step 4: fetch price bars
    print("[cycle] 3/6  Fetching Alpaca price bars...")
    bars = get_historical_bars(symbol, timeframe=TIMEFRAME, limit=BAR_LIMIT)

    # Steps 5-7: build MDPs with correct tensor dimensions
    print(f"[cycle] 4/6  Building MDPs "
          f"(sent:{n_sent}states  vol:{n_vol}states  regime:{N_REGIME}states)...")
    mdp_sent = build_l1_sentiment_mdp(llm_params)
    mdp_vol  = build_l1_volatility_mdp(llm_params)
    mdp_l2   = build_l2_mdp(
        mdp_sent, mdp_vol, llm_params,
        pref_profit=pref_profit, pref_loss=pref_loss,
        T=T, N_regime=N_REGIME,
    )
    mdp_l2 = inject_price_observations(mdp_l2, bars, T=T)

    # Step 8: restore Dirichlet params — L1 into mdp_sent/mdp_vol,
    #         L2 into mdp_l2 (skipped if N changed this cycle)
    mdp_sent, mdp_vol, mdp_l2 = _load_and_resize_params(
        mdp_l2, mdp_sent, mdp_vol, llm_params
    )

    # Step 9: run VB inference
    print("[cycle] 5/6  Running spm_MDP_VB_X...")
    result_mdp = spm_MDP_VB_X(mdp_l2)

    # Step 10: save updated Dirichlet params — L1 and L2 separately
    save_params(mdp_sent, mdp_vol, result_mdp)

    # Extract signal — uses actual array sizes from result_mdp
    signal = beliefs_to_signal(result_mdp, llm_params)

    # Step 11: execute signal on Alpaca
    print("[cycle] 6/6  Executing signal...")
    exec_result = execute_signal(signal, symbol=symbol, qty=QTY)

    # Step 12: log trade to trade_log.jsonl
    order      = exec_result.get("order") if exec_result["executed"] else None
    order_id   = order["id"] if order else None
    entry_price = float(order.get("filled_avg_price") or bars[-1]["close"]) \
                  if order else None

    log_trade(
        symbol        = symbol,
        action        = signal["action"],
        qty           = QTY,
        confidence    = signal["confidence"],
        regime_belief = signal["regime_belief"],
        sent_belief   = signal["sent_belief"],
        vol_belief    = signal["vol_belief"],
        llm_reasoning = llm_params["reasoning"],
        executed      = exec_result["executed"],
        order_id      = order_id,
        entry_price   = entry_price,
        skip_reason   = exec_result.get("reason") if not exec_result["executed"] else None,
    )

    # Step 14: Telegram report
    report = _format_report(symbol, signal, llm_params, exec_result, reflected)
    print(f"\n{report}")
    _send_telegram(report)


# ===========================================================================
# Main loop
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("Heartbeat — Active Inference Trading (variable-N memory)")
    print("=" * 60)
    print(f"  Symbol         : {HEARTBEAT_SYMBOL}")
    print(f"  Interval       : {HEARTBEAT_INTERVAL}s ({HEARTBEAT_INTERVAL//60} min)")
    print(f"  Market hours   : {'US only' if MARKET_HOURS_ONLY else '24/7'}")
    print(f"  Min confidence : {MIN_CONFIDENCE}")
    print(f"  Max N states   : 8 per factor")
    print("=" * 60)

    _send_telegram(
        f"🚀 Heartbeat started — {HEARTBEAT_SYMBOL}\n"
        f"Interval: {HEARTBEAT_INTERVAL//60} min  |  "
        f"Dynamic hidden states: active"
    )

    while True:
        time.sleep(HEARTBEAT_INTERVAL)

        if not _is_market_open():
            print(f"[heartbeat] Closed — "
                  f"{datetime.now().strftime('%H:%M:%S')} UTC")
            continue

        try:
            run_cycle(HEARTBEAT_SYMBOL)
        except Exception as e:
            msg = f"⚠️ Cycle error ({HEARTBEAT_SYMBOL}):\n{e}"
            print(f"[heartbeat] {msg}")
            _send_telegram(msg)


if __name__ == "__main__":
    main()
