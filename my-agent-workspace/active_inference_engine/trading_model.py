"""
trading_model.py  (LLM-driven redesign)
========================================
Active inference generative model for algorithmic trading.

Architecture
------------
Layer 1a — Sentiment MDP
    Observations : news headlines + macro text  (from web_search_tool)
    Hidden states: bullish / bearish / neutral   (3 states)
    A, D prior   : estimated by LLM API call     (JSON response)

Layer 1b — Volatility MDP
    Observations : same text corpus as L1a
    Hidden states: high / normal / low volatility (3 states)
    A, D prior   : estimated by LLM API call      (JSON response)

Layer 2  — Market Regime MDP
    Observations : discretised Alpaca price bars  (bull/bear/doji)
                 + soft L1a sentiment posterior
                 + soft L1b volatility posterior
    Hidden states: risk-on bull / risk-off bear / uncertain / trending  (4 states)
    Policies     : buy / hold / sell  (3 policies)
    Action       : place_market_order() via Alpaca if confidence >= threshold

All engine files (solver, inference, learning, utils, mdp_model, run_trials)
are completely unchanged.
"""

import copy
import json
import numpy as np
import matplotlib.pyplot as plt

from .utils     import spm_softmax, spm_norm
from .mdp_model import MDPModel
from .solver    import spm_MDP_VB_X

import sys, os
_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from tools.alpaca_tools    import get_historical_bars, place_market_order
from tools.web_search_tool import run_web_search

# Variable-N MDP builders — used by run_trading_step and TRADING_TOOLS
from memory.model_builder import (
    build_l1_sentiment_mdp  as _build_l1_sent,
    build_l1_volatility_mdp as _build_l1_vol,
    build_l2_mdp            as _build_l2,
    inject_price_observations as _inject_obs,
    beliefs_to_signal       as _beliefs_to_signal,
    bars_to_regime,          # lives in model_builder to avoid circular import
)


# ===========================================================================
# Hyperparameters
# ===========================================================================

NI             = 16
TAU            = 4.0
ALPHA          = 512.0
BETA           = 1.0
ERP            = 1.0
ETA            = 0.5
OMEGA          = 0.98
CONCENTRATION  = 64.0
T              = 8
N_POLICIES     = 3
D_FREEZE_SCALE = 100.0

SYMBOL         = "SPY"
TIMEFRAME      = "15Min"
BAR_LIMIT      = 50
QTY            = 1.0
MIN_CONFIDENCE = 0.65

PREF_PROFIT    =  2.0
PREF_LOSS      = -3.0
PREF_FLAT      =  0.0

LLM_MODEL      = "claude-sonnet-4-6"

N_REGIME       = 4
N_TIME         = T
N_POSITION     = 3
N_MODALITIES   = 4


# ===========================================================================
# Step 1 — Fetch Layer 1 observations via web search
# ===========================================================================

def fetch_l1_observations(symbol: str) -> str:
    """
    Collect raw text corpus for Layer 1 from three web searches:
        1. Recent news headlines for the symbol
        2. Macro environment (Fed, CPI, yield curve)
        3. Market volatility / risk sentiment (VIX, fear/greed)
    Returns a single concatenated string passed to the LLM estimator.
    """
    news  = run_web_search(f"{symbol} stock news today",           max_results=4)
    macro = run_web_search("Fed interest rates CPI inflation today", max_results=3)
    vix   = run_web_search("VIX volatility fear greed index today",  max_results=3)
    return f"=== NEWS ===\n{news}\n\n=== MACRO ===\n{macro}\n\n=== VOLATILITY ===\n{vix}"


# ===========================================================================
# Step 2 — LLM API call: estimate A matrices and D priors for L1
# ===========================================================================

_L1_PROMPT = """You are a quantitative analyst building an active inference \
generative model for trading {symbol}.

Below is today's market information:

{corpus}

Estimate TWO sets of generative model parameters from this text.

SENTIMENT factor hidden states:
  0 = bullish   1 = bearish   2 = neutral

A_sentiment[i][j] = P(observation i | hidden state j)
Observations: 0=clearly_positive  1=clearly_negative  2=mixed_or_neutral

D_sentiment[j] = prior probability of state j right now.

VOLATILITY factor hidden states:
  0 = high_volatility   1 = normal_volatility   2 = low_volatility

A_volatility[i][j] = P(observation i | hidden state j)
Observations: 0=high_stress_signals  1=normal_signals  2=calm_signals

D_volatility[j] = prior probability of that volatility state now.

Rules:
- Every column of every A matrix must sum to 1.0
- Every D vector must sum to 1.0
- All values between 0.0 and 1.0
- Base estimates on the text above

Respond ONLY with this exact JSON (no markdown, no extra text):
{{"A_sentiment":[[f,f,f],[f,f,f],[f,f,f]],"D_sentiment":[f,f,f],\
"A_volatility":[[f,f,f],[f,f,f],[f,f,f]],"D_volatility":[f,f,f],\
"reasoning":"one sentence"}}"""


def call_llm_for_l1(corpus: str, symbol: str) -> dict:
    """
    Call the Anthropic API to estimate L1 generative model parameters.
    Returns dict with keys: A_sentiment, D_sentiment, A_volatility,
    D_volatility, reasoning.
    Falls back to uninformative uniform matrices on any error.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = _L1_PROMPT.format(symbol=symbol, corpus=corpus[:6000])

    try:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.content[0].text.strip())

        # Normalise: clip negatives, column-normalise A matrices
        for key in ("A_sentiment", "A_volatility"):
            A = np.clip(np.array(data[key], dtype=float), 1e-6, None)
            data[key] = spm_norm(A).tolist()
        for key in ("D_sentiment", "D_volatility"):
            D = np.clip(np.array(data[key], dtype=float), 1e-6, None)
            data[key] = (D / D.sum()).tolist()

        # Always set n_sent / n_vol from actual A matrix shape
        data["n_sent"] = len(data["D_sentiment"])
        data["n_vol"]  = len(data["D_volatility"])
        data.setdefault("sent_names", [f"s{i}" for i in range(data["n_sent"])])
        data.setdefault("vol_names",  [f"v{i}" for i in range(data["n_vol"])])

        print(f"[L1 LLM] {data['reasoning']}")
        return data

    except Exception as e:
        print(f"[L1 LLM] Error ({e}) — using uniform fallback")
        u = [[1/3]*3]*3
        return {"A_sentiment": u, "D_sentiment": [1/3]*3,
                "A_volatility": u, "D_volatility": [1/3]*3,
                "n_sent": 3, "n_vol": 3,
                "sent_names": ["bullish", "bearish", "neutral"],
                "vol_names":  ["high", "normal", "low"],
                "reasoning": f"fallback ({e})"}


# ===========================================================================
# Step 3 — Build Layer 1 MDPs from LLM estimates
# ===========================================================================

def build_l1_sentiment_mdp(llm_params: dict) -> MDPModel:
    """
    Layer 1a — sentiment MDP.
    A and D come from LLM. B is identity (single timestep, no transition).
    """
    A_mat = np.array(llm_params["A_sentiment"],  dtype=float)
    D_vec = np.array(llm_params["D_sentiment"],   dtype=float)
    return MDPModel(
        A=[A_mat], B=[np.eye(3)[:, :, np.newaxis]], D=[D_vec], T=1,
        a=[A_mat * CONCENTRATION], d=[D_vec.copy()],
        alpha=ALPHA, beta=BETA, tau=TAU, erp=ERP, eta=ETA, omega=OMEGA, ni=NI,
    )


def build_l1_volatility_mdp(llm_params: dict) -> MDPModel:
    """
    Layer 1b — volatility MDP. Same structure as L1a, different A and D.
    """
    A_mat = np.array(llm_params["A_volatility"], dtype=float)
    D_vec = np.array(llm_params["D_volatility"],  dtype=float)
    return MDPModel(
        A=[A_mat], B=[np.eye(3)[:, :, np.newaxis]], D=[D_vec], T=1,
        a=[A_mat * CONCENTRATION], d=[D_vec.copy()],
        alpha=ALPHA, beta=BETA, tau=TAU, erp=ERP, eta=ETA, omega=OMEGA, ni=NI,
    )


# ===========================================================================
# Step 5 — Build combined L1 sub-MDP for solver's hierarchical link
# ===========================================================================

def _build_combined_l1(mdp_sent: MDPModel, mdp_vol: MDPModel) -> MDPModel:
    """
    Wrap both L1 MDPs into a single two-factor MDP so the solver's existing
    hierarchical link mechanism can route both posteriors to L2.

    Factor 0 = sentiment (3 states)
    Factor 1 = volatility (3 states)

    Each modality is an identity readout of one factor, so the solver's
    sub-MDP posterior directly becomes the soft L2 observation.
    """
    # A[0]: P(sent_obs | sent_state, vol_state) — identity on factor 0
    A_sent = np.zeros((3, 3, 3))
    for i in range(3):
        A_sent[i, i, :] = 1.0

    # A[1]: P(vol_obs | sent_state, vol_state) — identity on factor 1
    A_vol = np.zeros((3, 3, 3))
    for i in range(3):
        A_vol[i, :, i] = 1.0

    return MDPModel(
        A=[A_sent, A_vol],
        B=[np.eye(3)[:, :, np.newaxis],
           np.eye(3)[:, :, np.newaxis]],
        D=[mdp_sent.D[0].copy(), mdp_vol.D[0].copy()],
        T=1,
        a=[np.ones((3, 3, 3)) * CONCENTRATION,   # sentiment a: (No, Ns_sent, Ns_vol)
           np.ones((3, 3, 3)) * CONCENTRATION],   # volatility a: same shape
        d=[mdp_sent.d[0].copy(), mdp_vol.d[0].copy()],
        V=np.ones((0, 1, 2), dtype=int),
        alpha=ALPHA, beta=BETA, tau=TAU, erp=ERP, eta=ETA, omega=OMEGA, ni=NI,
    )


# ===========================================================================
# Step 6 — Build Layer 2 MDP (price bars + L1 LLM beliefs)
# ===========================================================================

def build_l2_mdp(mdp_sent: MDPModel,
                 mdp_vol:  MDPModel,
                 llm_params: dict) -> MDPModel:
    """
    Layer 2 — market regime MDP.

    4 observation modalities:
        0 = price bar (3)    ← Alpaca, injected via inject_price_observations
        1 = sentiment (3)    ← from L1a via hierarchical link
        2 = volatility (3)   ← from L1b via hierarchical link
        3 = feedback (3)     ← profitable/loss/flat at last bar

    3 hidden state factors:
        0 = market regime  (4): risk-on bull / risk-off bear / uncertain / trending
        1 = time in window (T)
        2 = position       (3): flat / long / short
    """
    D_sent = np.array(llm_params["D_sentiment"])
    D_vol  = np.array(llm_params["D_volatility"])

    # ── Priors ───────────────────────────────────────────────────────────
    D2 = [
        np.ones(N_REGIME) / N_REGIME,
        np.array([1.0] + [0.0] * (N_TIME - 1)),
        np.array([1.0, 0.0, 0.0]),
    ]
    d2 = [D2[0].copy(),
          D2[1].copy() * D_FREEZE_SCALE,
          D2[2].copy() * D_FREEZE_SCALE]

    # ── Likelihood matrices ───────────────────────────────────────────────
    # Modality 0: price bar (3) × regime (4)
    price_by_regime = spm_norm(np.array([
        [0.70, 0.10, 0.30, 0.50],   # bull bar
        [0.10, 0.70, 0.30, 0.30],   # bear bar
        [0.20, 0.20, 0.40, 0.20],   # doji
    ]))
    A2_price = np.zeros((3, N_REGIME, N_TIME, N_POSITION))
    for ti in range(N_TIME):
        for pi in range(N_POSITION):
            A2_price[:, :, ti, pi] = price_by_regime

    # Modality 1: sentiment (3) × regime (4), blended with LLM D_sent
    sent_by_regime = spm_norm(np.array([
        [0.70, 0.10, 0.40, 0.50],
        [0.10, 0.70, 0.30, 0.20],
        [0.20, 0.20, 0.30, 0.30],
    ]))
    sent_by_regime[:, 0] = spm_norm(
        (0.5 * sent_by_regime[:, 0] + 0.5 * D_sent).reshape(-1, 1)).ravel()
    A2_sentiment = np.zeros((3, N_REGIME, N_TIME, N_POSITION))
    for ti in range(N_TIME):
        for pi in range(N_POSITION):
            A2_sentiment[:, :, ti, pi] = sent_by_regime

    # Modality 2: volatility (3) × regime (4), blended with LLM D_vol
    vol_by_regime = spm_norm(np.array([
        [0.30, 0.60, 0.70, 0.40],
        [0.40, 0.30, 0.20, 0.40],
        [0.30, 0.10, 0.10, 0.20],
    ]))
    vol_by_regime[:, 2] = spm_norm(
        (0.5 * vol_by_regime[:, 2] + 0.5 * D_vol).reshape(-1, 1)).ravel()
    A2_volatility = np.zeros((3, N_REGIME, N_TIME, N_POSITION))
    for ti in range(N_TIME):
        for pi in range(N_POSITION):
            A2_volatility[:, :, ti, pi] = vol_by_regime

    # Modality 3: position feedback — only meaningful at last bar
    A2_feedback = np.zeros((3, N_REGIME, N_TIME, N_POSITION))
    for regime in range(N_REGIME):
        for ti in range(N_TIME - 1):
            for pi in range(N_POSITION):
                A2_feedback[0, regime, ti, pi] = 1.0
    tl = N_TIME - 1
    # long position
    A2_feedback[:, 0, tl, 1] = [0.0, 0.85, 0.15]
    A2_feedback[:, 1, tl, 1] = [0.0, 0.15, 0.85]
    A2_feedback[:, 2, tl, 1] = [1.0, 0.00, 0.00]
    A2_feedback[:, 3, tl, 1] = [0.0, 0.60, 0.40]
    # short position
    A2_feedback[:, 0, tl, 2] = [0.0, 0.15, 0.85]
    A2_feedback[:, 1, tl, 2] = [0.0, 0.85, 0.15]
    A2_feedback[:, 2, tl, 2] = [1.0, 0.00, 0.00]
    A2_feedback[:, 3, tl, 2] = [0.0, 0.50, 0.50]
    # flat position
    for regime in range(N_REGIME):
        A2_feedback[:, regime, tl, 0] = [1.0, 0.0, 0.0]

    A2 = [A2_price, A2_sentiment, A2_volatility, A2_feedback]
    a2 = [A * CONCENTRATION for A in A2]

    # ── Transitions ───────────────────────────────────────────────────────
    B2_regime = np.array([
        [0.80, 0.05, 0.10, 0.10],
        [0.05, 0.80, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
        [0.05, 0.05, 0.10, 0.70],
    ])[:, :, np.newaxis]

    B2_time = np.zeros((N_TIME, N_TIME, 1))
    for i in range(N_TIME - 1):
        B2_time[i + 1, i, 0] = 1.0
    B2_time[N_TIME - 1, N_TIME - 1, 0] = 1.0

    B2_pos = np.zeros((N_POSITION, N_POSITION, N_POLICIES))
    B2_pos[:, :, 0] = np.eye(N_POSITION)
    B2_pos[:, :, 1] = [[0,0,0],[1,1,1],[0,0,0]]
    B2_pos[:, :, 2] = [[0,0,0],[0,0,0],[1,1,1]]

    B2 = [B2_regime, B2_time, B2_pos]

    # ── Policies ─────────────────────────────────────────────────────────
    V2 = np.ones((T - 1, N_POLICIES, 3), dtype=int)
    V2[T - 2, 0, 2] = 1   # hold
    V2[T - 2, 1, 2] = 2   # buy
    V2[T - 2, 2, 2] = 3   # sell

    # ── Preferences ───────────────────────────────────────────────────────
    C2_fb = np.zeros((3, T))
    C2_fb[0, T - 1] = PREF_FLAT
    C2_fb[1, T - 1] = PREF_PROFIT
    C2_fb[2, T - 1] = PREF_LOSS
    C2 = [np.zeros((3, T)), np.zeros((3, T)), np.zeros((3, T)), C2_fb]

    # ── Hierarchical link ─────────────────────────────────────────────────
    # Combined L1 sub-MDP: factor 0=sentiment → modality 1, factor 1=vol → modality 2
    mdp_combined = _build_combined_l1(mdp_sent, mdp_vol)
    link = np.array([
        [0, 1, 0, 0],   # sentiment factor → L2 modality 1
        [0, 0, 1, 0],   # volatility factor → L2 modality 2
    ])

    return MDPModel(
        A=A2, B=B2, D=D2, T=T,
        a=a2, d=d2, C=C2, V=V2,
        alpha=ALPHA, beta=BETA, tau=TAU, erp=ERP,
        eta=ETA, omega=OMEGA, ni=NI,
        MDP=mdp_combined, link=link,
    )


# ===========================================================================
# Step 7 — Inject Alpaca price observations into L2 modality 0
# ===========================================================================

def inject_price_observations(mdp: MDPModel, bars: list[dict]) -> MDPModel:
    """
    Set true observations for modality 0 (price bar) from Alpaca bars.
    L1 posteriors are injected automatically by the solver via the link matrix.
    """
    regimes = bars_to_regime(bars[-T:])

    s = np.ones((3, T), dtype=int)
    s[1, :] = np.arange(1, T + 1)

    o = np.ones((N_MODALITIES, T), dtype=int)
    for t_idx, regime in enumerate(regimes[:T]):
        o[0, t_idx] = regime + 1

    mdp.s = s
    mdp.o = o
    return mdp


# ===========================================================================
# Step 8 — Extract signal from solved L2 MDP
# ===========================================================================

def beliefs_to_signal(mdp_result: MDPModel) -> dict:
    """Extract trading signal from solved L2 MDP."""
    policy_post  = mdp_result.R[:, -1]
    best_policy  = int(np.argmax(policy_post))
    confidence   = float(policy_post[best_policy])
    regime_belief = mdp_result.X[0][:, -1]

    sent_belief = np.ones(3) / 3
    vol_belief  = np.ones(3) / 3
    if mdp_result.mdp_t is not None:
        last_sub = next(
            (r for r in reversed(mdp_result.mdp_t) if r is not None), None)
        if last_sub is not None and last_sub.X is not None:
            if len(last_sub.X) > 0: sent_belief = last_sub.X[0][:, 0]
            if len(last_sub.X) > 1: vol_belief  = last_sub.X[1][:, 0]

    return {
        "action":        {0: "hold", 1: "buy", 2: "sell"}[best_policy],
        "confidence":    confidence,
        "regime_belief": regime_belief,
        "sent_belief":   sent_belief,
        "vol_belief":    vol_belief,
        "policy_post":   policy_post,
    }


# ===========================================================================
# Step 9 — Full trading step
# ===========================================================================

def run_trading_step(symbol: str = SYMBOL) -> dict:
    """
    One complete active inference cycle:
        1. Web search  → news + macro corpus
        2. LLM API     → A matrices + D priors for L1a and L1b
        3. Build L1 sentiment + volatility MDPs
        4. Fetch Alpaca price bars
        5. Build L2 regime MDP
        6. Inject price observations
        7. spm_MDP_VB_X  → posterior beliefs
        8. Extract signal
    """
    print(f"\n[trading_step] {'─'*40} {symbol}")
    print("[trading_step] 1/4  Web search (news + macro + VIX)...")
    corpus = fetch_l1_observations(symbol)

    print("[trading_step] 2/4  LLM estimating L1 A matrices and D priors...")
    llm_params = call_llm_for_l1(corpus, symbol)

    print("[trading_step] 3/4  Fetching price bars from Alpaca...")
    bars = get_historical_bars(symbol, timeframe=TIMEFRAME, limit=BAR_LIMIT)

    print("[trading_step] 4/4  Building MDPs and running VB inference...")
    mdp_sent = _build_l1_sent(llm_params)
    mdp_vol  = _build_l1_vol(llm_params)
    mdp_l2   = _build_l2(mdp_sent, mdp_vol, llm_params)
    mdp_l2   = _inject_obs(mdp_l2, bars, T=T)
    result   = spm_MDP_VB_X(mdp_l2)
    signal   = _beliefs_to_signal(result, llm_params)

    names = ["risk-on bull", "risk-off bear", "uncertain", "trending"]
    top   = names[int(np.argmax(signal["regime_belief"]))]
    n_sent = llm_params.get("n_sent", 3)
    n_vol  = llm_params.get("n_vol",  3)
    sent_names = signal.get("sent_names", [f"s{i}" for i in range(n_sent)])
    vol_names  = signal.get("vol_names",  [f"v{i}" for i in range(n_vol)])
    print(f"[trading_step] Signal  : {signal['action'].upper()} "
          f"(conf={signal['confidence']:.2f})  regime={top}")
    print(f"[trading_step] Sentiment ({n_sent} states): "
          + "  ".join(f"{n}={v:.2f}" for n, v in zip(sent_names, signal["sent_belief"])))
    print(f"[trading_step] Volatility ({n_vol} states): "
          + "  ".join(f"{n}={v:.2f}" for n, v in zip(vol_names, signal["vol_belief"])))
    print(f"[trading_step] LLM: {llm_params['reasoning']}")

    return {**signal, "bars": bars, "llm_params": llm_params, "mdp_result": result}


def execute_signal(signal: dict, symbol: str = SYMBOL, qty: float = QTY) -> dict:
    """Execute signal on Alpaca if confidence >= MIN_CONFIDENCE."""
    if signal["confidence"] < MIN_CONFIDENCE:
        return {"executed": False, "reason": "low_confidence", "signal": signal}
    if signal["action"] == "hold":
        return {"executed": False, "reason": "hold_signal", "signal": signal}
    try:
        order = place_market_order(symbol=symbol, qty=qty, side=signal["action"])
        return {"executed": True, "order": order, "signal": signal}
    except Exception as e:
        return {"executed": False, "reason": str(e), "signal": signal}


# ===========================================================================
# Upsonic @tool wrappers
# ===========================================================================

try:
    from upsonic.tools import tool

    @tool
    def active_inference_signal(symbol: str = SYMBOL) -> str:
        """
        Run the full active inference pipeline on live market data.
        Fetches news via web search, calls LLM to estimate generative model
        parameters for sentiment and volatility, fetches Alpaca price bars,
        runs hierarchical variational inference, and returns a trading signal.

        Args:
            symbol: Ticker symbol, e.g. "SPY", "AAPL", "QQQ"
        Returns:
            Signal with action, confidence, regime, sentiment and volatility beliefs.
        """
        step  = run_trading_step(symbol)
        names = ["risk-on bull", "risk-off bear", "uncertain", "trending"]
        top   = names[int(np.argmax(step["regime_belief"]))]
        lp    = step["llm_params"]
        sent_str = "  ".join(
            f"{n}={v:.2f}" for n, v in
            zip(step.get("sent_names", []), step["sent_belief"])
        )
        vol_str = "  ".join(
            f"{n}={v:.2f}" for n, v in
            zip(step.get("vol_names", []), step["vol_belief"])
        )
        return (
            f"Active inference signal for {symbol}:\n"
            f"  Action     : {step['action'].upper()}\n"
            f"  Confidence : {step['confidence']:.2f}\n"
            f"  Regime     : {top}\n"
            f"  Sentiment  : {sent_str} ({lp.get('n_sent',3)} states)\n"
            f"  Volatility : {vol_str} ({lp.get('n_vol',3)} states)\n"
            f"  LLM note   : {lp['reasoning']}"
        )

    @tool
    def active_inference_trade(symbol: str = SYMBOL, qty: float = QTY) -> str:
        """
        Run the active inference pipeline and execute a trade on Alpaca
        if confidence is above the minimum threshold.

        Args:
            symbol: Ticker symbol
            qty:    Number of shares
        Returns:
            Execution result or reason for not trading.
        """
        step   = run_trading_step(symbol)
        result = execute_signal(step, symbol=symbol, qty=qty)
        if result["executed"]:
            return (f"Executed {result['signal']['action'].upper()} {qty} {symbol} "
                    f"— order {result['order']['id']}. "
                    f"LLM: {step['llm_params']['reasoning']}")
        return (f"No trade for {symbol}. Reason: {result['reason']}. "
                f"Signal: {result['signal']['action']} "
                f"(conf={result['signal']['confidence']:.2f}). "
                f"LLM: {step['llm_params']['reasoning']}")

    TRADING_TOOLS = [active_inference_signal, active_inference_trade]

except ImportError:
    TRADING_TOOLS = []


# ===========================================================================
# Standalone dry-run:  python -m active_inference_engine.trading_model
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Active Inference Trading — dry run (no real orders placed)")
    print("=" * 60)
    step = run_trading_step(SYMBOL)
    print("\nDone. No orders submitted in dry-run mode.")
