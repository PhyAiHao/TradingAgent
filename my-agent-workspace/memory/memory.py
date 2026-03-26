"""
memory.py
=========
Three-layer memory system for the active inference trading agent.

Layer 1 — Trade log        : raw facts appended to trade_log.jsonl
Layer 2 — Parameter store  : learned Dirichlet a, d, e persisted to params.pkl
Layer 3 — LLM reflection   : trade history fed back to LLM to enrich hidden states

The LLM reflection layer is the answer to insufficient hidden state information.
The MDP can only track 3-4 discrete states, but markets are driven by many latent
factors (liquidity, institutional flow, macro regime, sentiment shifts). The LLM
reads past trade outcomes and reasons backwards — inferring what hidden conditions
must have been present — then updates the A matrix and D prior accordingly.
This enriches the generative model beyond what the hand-defined states can capture.
"""

import os
import sys
import json
import pickle
import pathlib
import numpy as np
from datetime import datetime
from typing import Optional
from active_inference_engine.utils import spm_norm

# ── Storage paths (inside the workspace memory/ folder) ──────────────────
# memory.py lives in my-agent-workspace/memory/ — workspace root is one level up
_WORKSPACE   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MEMORY_DIR  = pathlib.Path(_WORKSPACE) / "memory" / "trading"
_LOG_FILE    = _MEMORY_DIR / "trade_log.jsonl"
_PARAMS_FILE = _MEMORY_DIR / "dirichlet_params.pkl"

_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# Ensure workspace root is on sys.path so config and tools are importable
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from config import LLM_MODEL, REFLECTION_WINDOW, MIN_STATES, MAX_STATES

# How many past trades to include in the LLM reflection prompt
REFLECTION_WINDOW = int(os.getenv("REFLECTION_WINDOW", "10"))

# Hard limits on number of hidden states LLM can propose
MIN_STATES = 2
MAX_STATES = 8


# ===========================================================================
# Layer 1 — Trade log
# ===========================================================================

def log_trade(
    symbol:         str,
    action:         str,
    qty:            float,
    confidence:     float,
    regime_belief:  np.ndarray,
    sent_belief:    np.ndarray,
    vol_belief:     np.ndarray,
    llm_reasoning:  str,
    executed:       bool,
    order_id:       Optional[str]  = None,
    entry_price:    Optional[float] = None,
    skip_reason:    Optional[str]  = None,
) -> None:
    """
    Append one trade record to trade_log.jsonl.
    Called immediately after execute_signal() in heartbeat.py.
    """
    record = {
        "timestamp":      datetime.utcnow().isoformat(),
        "symbol":         symbol,
        "action":         action,
        "qty":            qty,
        "confidence":     round(float(confidence), 4),
        "executed":       executed,
        "order_id":       order_id,
        "entry_price":    entry_price,
        "skip_reason":    skip_reason,
        "regime_belief":  [round(float(x), 4) for x in regime_belief],
        "sent_belief":    [round(float(x), 4) for x in sent_belief],
        "vol_belief":     [round(float(x), 4) for x in vol_belief],
        "llm_reasoning":  llm_reasoning,
        # Outcome fields — filled in later by update_trade_outcome()
        "exit_price":     None,
        "pnl":            None,
        "outcome":        None,   # "profit" | "loss" | "flat"
    }
    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[memory] Trade logged: {action.upper()} {symbol} "
          f"(executed={executed})")


def update_trade_outcome(
    order_id:   str,
    exit_price: float,
    pnl:        float,
) -> None:
    """
    Update the outcome fields of a previously logged trade.
    Called by heartbeat.py after checking the Alpaca position P&L.
    """
    if not _LOG_FILE.exists():
        return

    records = []
    updated = False
    with open(_LOG_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("order_id") == order_id and r["outcome"] is None:
                r["exit_price"] = round(exit_price, 4)
                r["pnl"]        = round(pnl, 4)
                r["outcome"]    = "profit" if pnl > 0 else ("loss" if pnl < 0 else "flat")
                updated         = True
            records.append(r)

    if updated:
        with open(_LOG_FILE, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[memory] Outcome updated for order {order_id}: "
              f"P&L={pnl:+.2f}")


def load_recent_trades(n: int = REFLECTION_WINDOW) -> list[dict]:
    """Return the last n trades from the log (most recent last)."""
    if not _LOG_FILE.exists():
        return []
    with open(_LOG_FILE) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records[-n:]


def trade_log_summary(trades: list[dict]) -> str:
    """
    Build a compact human-readable summary of recent trades for the LLM prompt.
    Focuses on: what signal fired, what the beliefs were, and what the outcome was.
    """
    if not trades:
        return "No previous trades recorded."

    lines = [f"Recent {len(trades)} trades (oldest first):"]
    for i, t in enumerate(trades, 1):
        outcome = t.get("outcome") or ("pending" if t["executed"] else t.get("skip_reason", "skipped"))
        pnl_str = f"  P&L={t['pnl']:+.2f}" if t.get("pnl") is not None else ""
        regime  = ["risk-on bull", "risk-off bear", "uncertain", "trending"]
        top_reg = regime[int(np.argmax(t["regime_belief"]))]
        lines.append(
            f"{i}. [{t['timestamp'][:16]}] {t['action'].upper()} {t['symbol']} "
            f"conf={t['confidence']:.2f}  regime={top_reg}  "
            f"sent=[{','.join(str(x) for x in t['sent_belief'])}]  "
            f"vol=[{','.join(str(x) for x in t['vol_belief'])}]  "
            f"outcome={outcome}{pnl_str}"
        )
        if t.get("llm_reasoning"):
            lines.append(f"   LLM at time: {t['llm_reasoning']}")
    return "\n".join(lines)


# ===========================================================================
# Layer 2 — Dirichlet parameter store
# ===========================================================================

def save_params(mdp_l1_sent, mdp_l1_vol, mdp_l2_result) -> None:
    """
    Persist learned Dirichlet concentration parameters to disk.

    Saves L1 (sentiment + volatility) and L2 params SEPARATELY so that
    resize_params always receives correctly shaped arrays.

    L1 a shapes: (No=3, N_sent) and (No=3, N_vol)  — 2D
    L2 a shapes: (No, N_regime, N_time, N_pos) × 4  — 4D
    """
    payload = {
        "saved_at": datetime.utcnow().isoformat(),
        # L1 sentiment MDP params
        "l1_sent_a":   mdp_l1_sent.a[0].copy()   if mdp_l1_sent.a  else None,
        "l1_sent_d":   mdp_l1_sent.d[0].copy()   if mdp_l1_sent.d  else None,
        "l1_sent_a_0": mdp_l1_sent.a_0[0].copy() if mdp_l1_sent.a_0 else None,
        "l1_sent_d_0": mdp_l1_sent.d_0[0].copy() if mdp_l1_sent.d_0 else None,
        # L1 volatility MDP params
        "l1_vol_a":    mdp_l1_vol.a[0].copy()    if mdp_l1_vol.a   else None,
        "l1_vol_d":    mdp_l1_vol.d[0].copy()    if mdp_l1_vol.d   else None,
        "l1_vol_a_0":  mdp_l1_vol.a_0[0].copy()  if mdp_l1_vol.a_0 else None,
        "l1_vol_d_0":  mdp_l1_vol.d_0[0].copy()  if mdp_l1_vol.d_0 else None,
        # L2 MDP params (4D arrays — saved as-is for L2 continuity)
        "l2_a":   [p.copy() for p in mdp_l2_result.a]   if mdp_l2_result.a   else None,
        "l2_d":   [p.copy() for p in mdp_l2_result.d]   if mdp_l2_result.d   else None,
        "l2_a_0": [p.copy() for p in mdp_l2_result.a_0] if mdp_l2_result.a_0 else None,
        "l2_d_0": [p.copy() for p in mdp_l2_result.d_0] if mdp_l2_result.d_0 else None,
        "l2_e":   mdp_l2_result.e.copy()                 if mdp_l2_result.e   else None,
        "l2_e_0": mdp_l2_result.e_0.copy()               if mdp_l2_result.e_0 else None,
    }
    _PARAMS_FILE.write_bytes(pickle.dumps(payload))
    l1s = mdp_l1_sent.a[0].shape if mdp_l1_sent.a else "?"
    l1v = mdp_l1_vol.a[0].shape  if mdp_l1_vol.a  else "?"
    print(f"[memory] Params saved — L1 sent:{l1s} vol:{l1v}")


def load_params_raw() -> dict | None:
    """Return raw saved payload dict, or None if no file exists."""
    if not _PARAMS_FILE.exists():
        return None
    return pickle.loads(_PARAMS_FILE.read_bytes())


# ===========================================================================
# Layer 3 — LLM reflection
# ===========================================================================

_REFLECTION_PROMPT = """You are a quantitative analyst reviewing past trades to \
improve a generative model for {symbol}.

=== RECENT TRADE HISTORY ===
{trade_summary}

=== TODAY'S MARKET INFORMATION ===
{corpus}

Your task: propose REVISED generative model parameters that account for BOTH \
today's information AND lessons from past trade outcomes.

CRITICAL — you may propose a DIFFERENT NUMBER OF HIDDEN STATES than before.
Look for patterns in the trade history where the same label (e.g. "bullish") \
preceded very different outcomes. That is evidence that the label is masking \
multiple distinct latent conditions that need separate states.

For example:
- If "bullish" sometimes led to profit and sometimes to loss, consider splitting \
it into "bullish_high_volume" and "bullish_low_volume" (or whatever latent factor \
explains the difference).
- If two states always have the same outcome, consider merging them.
- Add a new state ONLY if you can name at least 2 trades that would have been \
classified differently under the new state.

SENTIMENT factor:
  Propose N_sent states (integer, 2 ≤ N_sent ≤ 8).
  Name each state (short label, no spaces).
  A_sentiment shape: (3 rows of observations × N_sent columns of states)
    Row 0 = P(clearly_positive_news | state)
    Row 1 = P(clearly_negative_news | state)
    Row 2 = P(mixed_or_neutral_news | state)
  D_sentiment: prior probability of each state NOW (length N_sent, sums to 1).

VOLATILITY factor:
  Propose N_vol states (integer, 2 ≤ N_vol ≤ 8).
  Name each state.
  A_volatility shape: (3 rows × N_vol columns)
    Row 0 = P(high_stress_signals | state)
    Row 1 = P(normal_signals | state)
    Row 2 = P(calm_signals | state)
  D_volatility: prior NOW (length N_vol, sums to 1).

PREFERENCE update:
  pref_profit: how strongly to prefer profit (range 0.5 to 5.0, default 2.0)
  pref_loss: how strongly to avoid loss (range -8.0 to -0.5, default -3.0)
  Increase |pref_loss| if recent trades have been losing.

Rules:
- Every COLUMN of every A matrix must sum to 1.0
- Every D vector must sum to 1.0
- All values between 0.0 and 1.0

Respond ONLY with this exact JSON (no markdown, no extra text):
{{"n_sent": int, "sent_names": ["name0","name1",...], \
"A_sentiment": [[col sums to 1],...N_sent cols...], \
"D_sentiment": [f,...N_sent values...], \
"n_vol": int, "vol_names": ["name0","name1",...], \
"A_volatility": [[col sums to 1],...N_vol cols...], \
"D_volatility": [f,...N_vol values...], \
"pref_profit": f, "pref_loss": f, \
"reasoning": "one sentence explaining dominant lesson from trade history"}}"""


def llm_reflect(
    corpus:  str,
    symbol:  str,
    n_trades: int = REFLECTION_WINDOW,
) -> dict:
    """
    Layer 3: LLM reads trade history + today's news and returns revised
    A matrices, D priors, and preference updates.

    This is the key mechanism for addressing insufficient hidden states:
    the LLM reasons about what latent conditions caused past errors and
    encodes that understanding into the generative model parameters.

    Returns the same dict structure as call_llm_for_l1(), plus:
        pref_profit : float
        pref_loss   : float
        reasoning   : str
    Falls back to plain call_llm_for_l1() output if no trade history exists.
    """
    import anthropic

    trades        = load_recent_trades(n_trades)
    trade_summary = trade_log_summary(trades)

    # If no history yet, no reflection possible — return None so caller
    # falls back to the standard call_llm_for_l1()
    if not trades:
        print("[memory] No trade history yet — skipping reflection")
        return None

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = _REFLECTION_PROMPT.format(
        symbol        = symbol,
        trade_summary = trade_summary,
        corpus        = corpus[:4000],
    )

    try:
        resp = client.messages.create(
            model     = LLM_MODEL,
            max_tokens = 1024,
            messages  = [{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.content[0].text.strip())

        # Extract and validate N
        n_sent = int(np.clip(data.get("n_sent", 3), MIN_STATES, MAX_STATES))
        n_vol  = int(np.clip(data.get("n_vol",  3), MIN_STATES, MAX_STATES))
        data["n_sent"] = n_sent
        data["n_vol"]  = n_vol

        # Validate and normalise A matrices — shape (3 obs, N states)
        # LLM returns columns, so A[i] = column i (one per state)
        # We need shape (n_obs=3, n_states) — transpose if needed
        for key, n_states in (("A_sentiment", n_sent), ("A_volatility", n_vol)):
            raw = np.array(data[key], dtype=float)
            # LLM may return (n_states, 3) or (3, n_states) — normalise to (3, n_states)
            if raw.shape == (n_states, 3):
                raw = raw.T          # transpose to (3, n_states)
            elif raw.shape != (3, n_states):
                # Shape mismatch — rebuild as uniform
                print(f"[memory] {key} shape mismatch {raw.shape}, "
                      f"expected (3, {n_states}) — using uniform")
                raw = np.ones((3, n_states)) / 3
            raw = np.clip(raw, 1e-6, None)
            data[key] = spm_norm(raw).tolist()   # column-normalise → (3, n_states)

        for key, n_states in (("D_sentiment", n_sent), ("D_volatility", n_vol)):
            D = np.clip(np.array(data[key], dtype=float).ravel(), 1e-6, None)
            if len(D) != n_states:
                print(f"[memory] {key} length {len(D)} ≠ {n_states} — using uniform")
                D = np.ones(n_states)
            data[key] = (D / D.sum()).tolist()

        data["pref_profit"] = float(np.clip(data.get("pref_profit", 2.0),  0.5,  5.0))
        data["pref_loss"]   = float(np.clip(data.get("pref_loss",  -3.0), -8.0, -0.5))

        print(f"[memory] Reflection: n_sent={n_sent} n_vol={n_vol}  "
              f"{data['reasoning']}")
        if data.get("sent_names"):
            print(f"[memory]   Sentiment states: {data['sent_names']}")
        if data.get("vol_names"):
            print(f"[memory]   Volatility states: {data['vol_names']}")
        return data

    except Exception as e:
        print(f"[memory] Reflection failed ({e}) — falling back to standard L1 call")
        return None
