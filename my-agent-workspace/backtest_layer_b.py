"""
backtest_layer_b.py
===================
Layer B backtester — tests the active inference engine on historical
price bars using a single frozen set of LLM parameters.

This answers the question: "Does spm_MDP_VB_X produce signals that
are better than random on real historical data?" — without making one
LLM API call per bar (which would be expensive).

Run from workspace root:
    python3 backtest_layer_b.py
    python3 backtest_layer_b.py --symbol AAPL --start 2024-06-01 --end 2024-12-31
    python3 backtest_layer_b.py --fast          # skip LLM, use uniform priors

Output:
    - Console summary (P&L, win rate, Sharpe, drawdown)
    - backtest_results.csv  — one row per signal window
    - backtest_report.png   — equity curve + signal breakdown charts
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt

# ── Workspace root on sys.path ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    SYMBOL, TIMEFRAME, BAR_LIMIT, QTY, MIN_CONFIDENCE,
    T, N_REGIME, NI, TAU, ALPHA, BETA, ERP, ETA, OMEGA,
    CONCENTRATION, N_POLICIES, D_FREEZE_SCALE,
    PREF_PROFIT, PREF_LOSS, PREF_FLAT,
    LLM_MODEL,
)

from active_inference_engine.trading_model import (
    call_llm_for_l1,
)
from active_inference_engine.solver import spm_MDP_VB_X
from memory.model_builder import (
    build_l1_sentiment_mdp,
    build_l1_volatility_mdp,
    build_l2_mdp,
    inject_price_observations,
    beliefs_to_signal,
    bars_to_regime,
)


# ===========================================================================
# CLI arguments
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Layer B active inference backtest")
    p.add_argument("--symbol",  default=SYMBOL,      help="Ticker symbol")
    p.add_argument("--start",   default="2024-06-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",     default="2024-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--conf",    type=float, default=MIN_CONFIDENCE,
                   help="Confidence threshold for high-conf analysis")
    p.add_argument("--fast",    action="store_true",
                   help="Skip LLM call — use uniform priors (fastest)")
    p.add_argument("--out-csv", default="backtest_results.csv")
    p.add_argument("--out-png", default="backtest_report.png")
    return p.parse_args()


# ===========================================================================
# Step 1 — Fetch historical bars from Alpaca
# ===========================================================================

def fetch_bars(symbol: str, start: str, end: str) -> list[dict]:
    """
    Fetch 15-minute bars from Alpaca for the given date range.
    Returns list of dicts with keys: timestamp, open, high, low, close, volume.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests   import StockBarsRequest
    from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit

    print(f"[backtest] Fetching {symbol} bars {start} → {end} ...")
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    req = StockBarsRequest(
        symbol_or_symbols = symbol,
        timeframe         = TimeFrame(15, TimeFrameUnit.Minute),
        start             = datetime.strptime(start, "%Y-%m-%d"),
        end               = datetime.strptime(end,   "%Y-%m-%d"),
    )
    raw = client.get_stock_bars(req)[symbol]
    bars = [
        {
            "timestamp": str(b.timestamp),
            "open":      float(b.open),
            "high":      float(b.high),
            "low":       float(b.low),
            "close":     float(b.close),
            "volume":    int(b.volume),
        }
        for b in raw
    ]
    print(f"[backtest] Loaded {len(bars)} bars")
    return bars


# ===========================================================================
# Step 2 — Build frozen LLM params
# ===========================================================================

def build_llm_params(fast: bool, symbol: str) -> dict:
    """
    Layer B uses a SINGLE frozen set of LLM params for the entire backtest.
    This isolates the active inference engine from LLM variability.

    --fast: use uniform priors (no API call, 3 states each)
    default: one real LLM call with a neutral corpus
    """
    if fast:
        print("[backtest] Fast mode — using uniform priors (no LLM call)")
        u = [[1/3]*3]*3
        return {
            "A_sentiment":  u, "D_sentiment":  [1/3, 1/3, 1/3],
            "A_volatility": u, "D_volatility": [1/3, 1/3, 1/3],
            "n_sent": 3, "n_vol": 3,
            "sent_names": ["bullish", "bearish", "neutral"],
            "vol_names":  ["high",    "normal",  "low"],
            "reasoning": "uniform priors (fast mode)",
        }

    print(f"[backtest] Calling LLM for frozen L1 params ({symbol}) ...")
    corpus = (
        f"Historical backtest for {symbol}. "
        "Assume neutral market conditions as the base prior. "
        "Use balanced priors across bullish, bearish, and neutral states."
    )
    params = call_llm_for_l1(corpus, symbol)
    params.setdefault("n_sent", 3)
    params.setdefault("n_vol",  3)
    params.setdefault("sent_names", ["bullish", "bearish", "neutral"])
    params.setdefault("vol_names",  ["high",    "normal",  "low"])
    print(f"[backtest] LLM params: n_sent={params['n_sent']} "
          f"n_vol={params['n_vol']}  {params['reasoning']}")
    return params


# ===========================================================================
# Step 3 — Run one inference window
# ===========================================================================

def run_window(bars_window: list[dict], llm_params: dict) -> dict:
    """
    Run one complete active inference cycle on a window of bars.
    Returns signal dict from beliefs_to_signal.
    """
    mdp_sent = build_l1_sentiment_mdp(llm_params)
    mdp_vol  = build_l1_volatility_mdp(llm_params)
    mdp_l2   = build_l2_mdp(
        mdp_sent, mdp_vol, llm_params,
        pref_profit=PREF_PROFIT, pref_loss=PREF_LOSS,
        T=T, N_regime=N_REGIME,
    )
    mdp_l2  = inject_price_observations(mdp_l2, bars_window, T=T)
    result  = spm_MDP_VB_X(mdp_l2)
    return beliefs_to_signal(result, llm_params)


# ===========================================================================
# Step 4 — Sliding window backtest loop
# ===========================================================================

def run_backtest(all_bars: list[dict], llm_params: dict) -> pd.DataFrame:
    """
    Slide a window of (BAR_LIMIT + T) bars across the full history.

    For each position i:
      - Use bars[i-BAR_LIMIT : i+T] as input to the inference engine
      - Signal is evaluated at bar i (entry)
      - Exit is simulated T bars later (bar i+T)
      - P&L = direction × (exit_close - entry_close) × QTY

    Returns a DataFrame with one row per evaluated window.
    """
    needed = BAR_LIMIT + T
    total  = len(all_bars)

    if total < needed + T:
        raise ValueError(
            f"Not enough bars: need {needed + T}, got {total}. "
            "Widen your date range."
        )

    rows    = []
    n_wins  = total - needed - T + 1
    print(f"[backtest] Running {n_wins} inference windows ...")

    for idx, i in enumerate(range(needed, total - T)):
        if idx % 50 == 0:
            pct = idx / n_wins * 100
            print(f"[backtest]   {idx}/{n_wins}  ({pct:.0f}%)", end="\r")

        bars_window  = all_bars[i - BAR_LIMIT : i + T]
        entry_bar    = all_bars[i]
        exit_bar     = all_bars[i + T]

        try:
            signal = run_window(bars_window, llm_params)
        except Exception as e:
            # Skip windows where VB fails (e.g. degenerate observations)
            continue

        action    = signal["action"]
        conf      = signal["confidence"]
        direction = {"buy": 1, "sell": -1, "hold": 0}[action]
        pnl       = direction * (exit_bar["close"] - entry_bar["close"]) * QTY

        regime_names = ["risk-on bull", "risk-off bear", "uncertain", "trending"]
        top_regime   = regime_names[int(np.argmax(signal["regime_belief"]))]

        rows.append({
            "timestamp":    entry_bar["timestamp"],
            "entry_close":  entry_bar["close"],
            "exit_close":   exit_bar["close"],
            "action":       action,
            "confidence":   round(conf, 4),
            "pnl":          round(pnl, 4),
            "regime":       top_regime,
            "sent_belief":  signal["sent_belief"].tolist(),
            "vol_belief":   signal["vol_belief"].tolist(),
            "regime_belief":signal["regime_belief"].tolist(),
        })

    print()   # newline after progress bar
    return pd.DataFrame(rows)


# ===========================================================================
# Step 5 — Compute metrics
# ===========================================================================

def compute_metrics(df: pd.DataFrame, conf_threshold: float) -> dict:
    """
    Compute key performance metrics from the raw results DataFrame.
    Separates all trades from high-confidence trades.
    """
    trades      = df[df["action"] != "hold"].copy()
    high_conf   = trades[trades["confidence"] >= conf_threshold].copy()

    def _metrics(subset: pd.DataFrame, label: str) -> dict:
        if len(subset) == 0:
            return {f"{label}_n": 0}
        pnl         = subset["pnl"]
        cum_pnl     = pnl.cumsum()
        drawdown    = (cum_pnl - cum_pnl.cummax()).min()
        sharpe      = pnl.mean() / (pnl.std() + 1e-8)
        win_rate    = (pnl > 0).mean()
        buys        = (subset["action"] == "buy").sum()
        sells       = (subset["action"] == "sell").sum()
        return {
            f"{label}_n":           len(subset),
            f"{label}_total_pnl":   round(pnl.sum(), 2),
            f"{label}_avg_pnl":     round(pnl.mean(), 4),
            f"{label}_win_rate":    round(win_rate, 4),
            f"{label}_sharpe":      round(sharpe, 4),
            f"{label}_max_dd":      round(drawdown, 2),
            f"{label}_buys":        int(buys),
            f"{label}_sells":       int(sells),
        }

    m = {
        "total_windows": len(df),
        "holds":         int((df["action"] == "hold").sum()),
        **_metrics(trades,    "all"),
        **_metrics(high_conf, "hc"),
    }
    return m


# ===========================================================================
# Step 6 — Print report
# ===========================================================================

def print_report(m: dict, symbol: str, start: str, end: str,
                 conf: float, llm_params: dict) -> None:
    regime_names = ["risk-on bull", "risk-off bear", "uncertain", "trending"]
    print(f"\n{'='*60}")
    print(f"  Layer B Backtest — {symbol}  {start} → {end}")
    print(f"{'='*60}")
    print(f"  LLM reasoning : {llm_params['reasoning']}")
    print(f"  n_sent={llm_params['n_sent']}  n_vol={llm_params['n_vol']}")
    print(f"  T={T}  NI={NI}  N_regime={N_REGIME}  BAR_LIMIT={BAR_LIMIT}")
    print(f"  MIN_CONFIDENCE={conf}  QTY={QTY}")
    print(f"\n  ── All windows ──")
    print(f"  Windows evaluated : {m['total_windows']}")
    print(f"  Holds             : {m['holds']}")
    print(f"  Trades (all)      : {m['all_n']}  "
          f"(buy={m['all_buys']}  sell={m['all_sells']})")
    print(f"  Total P&L         : ${m['all_total_pnl']:+.2f}")
    print(f"  Avg P&L / trade   : ${m['all_avg_pnl']:+.4f}")
    print(f"  Win rate          : {m['all_win_rate']:.1%}")
    print(f"  Sharpe ratio      : {m['all_sharpe']:.3f}")
    print(f"  Max drawdown      : ${m['all_max_dd']:.2f}")
    print(f"\n  ── High-confidence (≥ {conf}) ──")
    print(f"  Trades (high-conf): {m['hc_n']}  "
          f"(buy={m.get('hc_buys',0)}  sell={m.get('hc_sells',0)})")
    print(f"  Total P&L         : ${m.get('hc_total_pnl', 0):+.2f}")
    print(f"  Win rate          : {m.get('hc_win_rate', 0):.1%}")
    print(f"  Sharpe ratio      : {m.get('hc_sharpe', 0):.3f}")
    print(f"  Max drawdown      : ${m.get('hc_max_dd', 0):.2f}")

    # Interpretation
    print(f"\n  ── Interpretation ──")
    hc_wr = m.get("hc_win_rate", 0)
    hc_sh = m.get("hc_sharpe",   0)
    if m["hc_n"] < 10:
        print("  ⚠️  Too few high-conf trades to draw conclusions.")
        print("     Lower MIN_CONFIDENCE or widen the date range.")
    elif hc_wr >= 0.55 and hc_sh >= 0.5:
        print("  ✅ High-conf signals look promising.")
        print("     Safe to run live — monitor first 20 trades carefully.")
    elif hc_wr >= 0.50:
        print("  ⚠️  Marginally above random.")
        print("     Consider tuning PREF_PROFIT/PREF_LOSS or MIN_CONFIDENCE.")
    else:
        print("  ❌ High-conf signals are below random.")
        print("     Do NOT run live. Review LLM params and N_REGIME.")
    print(f"{'='*60}\n")


# ===========================================================================
# Step 7 — Plot
# ===========================================================================

def plot_results(df: pd.DataFrame, conf: float, symbol: str,
                 out_path: str) -> None:
    trades    = df[df["action"] != "hold"].copy()
    high_conf = trades[trades["confidence"] >= conf].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Layer B Backtest — {symbol}  "
        f"({df['timestamp'].iloc[0][:10]} → {df['timestamp'].iloc[-1][:10]})",
        fontsize=13,
    )

    # ── Panel 1: equity curves ────────────────────────────────────────────
    ax = axes[0, 0]
    if len(trades):
        ax.plot(trades["pnl"].cumsum().values,
                color="#185FA5", linewidth=1.5, label="All trades")
    if len(high_conf):
        ax.plot(high_conf["pnl"].cumsum().values,
                color="#0F6E56", linewidth=2, label=f"High-conf (≥{conf})")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Equity curve (cumulative P&L)")
    ax.set_xlabel("Trade number")
    ax.set_ylabel(f"P&L (${QTY} per trade)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: confidence distribution ──────────────────────────────────
    ax = axes[0, 1]
    if len(trades):
        ax.hist(trades["confidence"], bins=20,
                color="#185FA5", alpha=0.7, label="All trades")
    ax.axvline(conf, color="#A32D2D", linewidth=1.5,
               linestyle="--", label=f"Threshold ({conf})")
    ax.set_title("Confidence distribution")
    ax.set_xlabel("Policy posterior confidence")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: P&L per trade coloured by action ─────────────────────────
    ax = axes[1, 0]
    colors = {"buy": "#0F6E56", "sell": "#A32D2D"}
    for action in ["buy", "sell"]:
        sub = trades[trades["action"] == action]
        if len(sub):
            ax.scatter(range(len(sub)), sub["pnl"].values,
                       c=colors[action], alpha=0.5, s=15, label=action)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("P&L per trade (buy=green, sell=red)")
    ax.set_xlabel("Trade index")
    ax.set_ylabel("P&L ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 4: action breakdown ─────────────────────────────────────────
    ax = axes[1, 1]
    action_counts = df["action"].value_counts()
    bar_colors = {"buy": "#0F6E56", "hold": "#888780", "sell": "#A32D2D"}
    bars = ax.bar(
        action_counts.index,
        action_counts.values,
        color=[bar_colors.get(a, "#185FA5") for a in action_counts.index],
    )
    for bar, val in zip(bars, action_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=10)
    ax.set_title("Signal breakdown")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[backtest] Chart saved → {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # Fetch data
    all_bars = fetch_bars(args.symbol, args.start, args.end)

    # Build frozen LLM params (one call or uniform)
    llm_params = build_llm_params(args.fast, args.symbol)

    # Run backtest
    df = run_backtest(all_bars, llm_params)

    if df.empty:
        print("[backtest] No results — check date range and bar availability.")
        return

    # Save CSV
    df.to_csv(args.out_csv, index=False)
    print(f"[backtest] Results saved → {args.out_csv}  ({len(df)} rows)")

    # Metrics and report
    m = compute_metrics(df, args.conf)
    print_report(m, args.symbol, args.start, args.end, args.conf, llm_params)

    # Plot
    plot_results(df, args.conf, args.symbol, args.out_png)


if __name__ == "__main__":
    main()
