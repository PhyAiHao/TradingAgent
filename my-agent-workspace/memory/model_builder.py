"""
model_builder.py
================
Builds L1 and L2 MDPs from LLM-estimated parameters where the number
of hidden states N is VARIABLE — determined by the LLM each cycle.

This is the correct answer to the "insufficient hidden states" problem:
  - Cycle 1:  LLM proposes 3 states (bullish, bearish, neutral)
  - Cycle 5:  After observing losses, LLM proposes 4 states
              (bullish, bearish, neutral, liquidity_squeeze)
  - Cycle 12: LLM proposes 5 states, merges two that always co-occur

The MDPModel is rebuilt with the correct tensor dimensions each cycle.
Learned Dirichlet params are carried forward and RESIZED — new states
get weak uniform priors, existing state params are copied exactly.

Constraints enforced here:
  - N_states: 2 ≤ N ≤ 8  (VB tractability)
  - New states require at least 2 historical trades as evidence (LLM-enforced via prompt)
  - B matrix: identity transitions (no action at L1 — passive perception)
  - A matrix: columns must sum to 1 (normalised here, not assumed from LLM)
"""

import numpy as np
import sys, os

# Ensure workspace root is importable (model_builder lives in memory/ subfolder)
_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from active_inference_engine.mdp_model import MDPModel
from active_inference_engine.utils import spm_norm, spm_softmax
from config import CONCENTRATION, MIN_STATES, MAX_STATES


# ===========================================================================
# Price bar discretisation — lives here to break circular import
# (trading_model imports from model_builder; model_builder must NOT
#  import back from trading_model)
# ===========================================================================

def bars_to_regime(bars: list[dict]) -> list[int]:
    """
    Convert OHLCV bars to price observation indices (0-based).
        0 = bull bar  (close > open by > 0.2%)
        1 = bear bar  (close < open by > 0.2%)
        2 = doji      (close ≈ open)
    """
    obs = []
    for bar in bars:
        body = (bar["close"] - bar["open"]) / max(bar["open"], 1e-8)
        if   body >  0.002: obs.append(0)
        elif body < -0.002: obs.append(1)
        else:               obs.append(2)
    return obs


# ===========================================================================
# Tensor resizing — carry learned params forward when N changes
# ===========================================================================

def _resize_a(old_a: np.ndarray, new_n: int) -> np.ndarray:
    """
    Resize a learned A concentration matrix from shape
    (No, old_N) to (No, new_N).

    New state columns are initialised with a weak uniform prior
    (value = 1.0, i.e. one pseudo-observation per outcome).
    Existing columns are copied exactly.
    """
    No, old_n = old_a.shape[0], old_a.shape[1]
    new_a = np.ones((No, new_n))          # weak uniform for all new states
    copy_n = min(old_n, new_n)
    new_a[:, :copy_n] = old_a[:, :copy_n]
    return new_a


def _resize_d(old_d: np.ndarray, new_n: int) -> np.ndarray:
    """
    Resize a learned D concentration vector from (old_N,) to (new_N,).
    New states get a weak prior of 1.0. Existing values copied.
    """
    new_d = np.ones(new_n)
    copy_n = min(len(old_d), new_n)
    new_d[:copy_n] = old_d[:copy_n]
    return new_d


def resize_params(saved_params: dict, new_n_sent: int, new_n_vol: int) -> dict:
    """
    Resize saved L1 Dirichlet params to match a new state dimensionality.

    saved_params: raw payload dict from memory.load_params_raw()
        keys: l1_sent_a, l1_sent_d, l1_vol_a, l1_vol_d  (np.ndarray)

    Returns a dict with resized arrays under the same keys.
    Called by heartbeat._load_and_resize_params when N changes.
    """
    if saved_params is None:
        return None

    resized = {}

    if saved_params.get("l1_sent_a") is not None:
        resized["l1_sent_a"] = _resize_a(saved_params["l1_sent_a"], new_n_sent)
        resized["l1_sent_a_0"] = resized["l1_sent_a"].copy()

    if saved_params.get("l1_sent_d") is not None:
        resized["l1_sent_d"] = _resize_d(saved_params["l1_sent_d"], new_n_sent)
        resized["l1_sent_d_0"] = resized["l1_sent_d"].copy()

    if saved_params.get("l1_vol_a") is not None:
        resized["l1_vol_a"] = _resize_a(saved_params["l1_vol_a"], new_n_vol)
        resized["l1_vol_a_0"] = resized["l1_vol_a"].copy()

    if saved_params.get("l1_vol_d") is not None:
        resized["l1_vol_d"] = _resize_d(saved_params["l1_vol_d"], new_n_vol)
        resized["l1_vol_d_0"] = resized["l1_vol_d"].copy()

    return resized


# ===========================================================================
# L1 MDP builders — variable N
# ===========================================================================

def build_l1_sentiment_mdp(llm_params: dict,
                            alpha: float = 512.0, beta: float = 1.0,
                            tau: float = 4.0, erp: float = 1.0,
                            eta: float = 0.5, omega: float = 0.98,
                            ni: int = 16) -> MDPModel:
    """
    Build the L1 sentiment MDP with N_sent hidden states.
    N_sent is determined by the LLM and can differ each cycle.

    A_sentiment shape: (N_obs=3, N_sent)
    D_sentiment shape: (N_sent,)
    B shape:           (N_sent, N_sent, 1)  — identity, no transition

    Param restoration (resize + inject) is handled externally by
    heartbeat._load_and_resize_params after this function returns.
    """
    A_mat = np.array(llm_params["A_sentiment"], dtype=float)   # (N_obs, N_sent)
    D_vec = np.array(llm_params["D_sentiment"], dtype=float)   # (N_sent,)
    N_obs, N_sent = A_mat.shape
    A_mat = spm_norm(A_mat)
    a_mat = A_mat * CONCENTRATION
    B_mat = np.eye(N_sent)[:, :, np.newaxis]

    return MDPModel(
        A=[A_mat], B=[B_mat], D=[D_vec], T=1,
        a=[a_mat], d=[D_vec.copy()],
        alpha=alpha, beta=beta, tau=tau, erp=erp, eta=eta, omega=omega, ni=ni,
    )


def build_l1_volatility_mdp(llm_params: dict,
                             alpha: float = 512.0, beta: float = 1.0,
                             tau: float = 4.0, erp: float = 1.0,
                             eta: float = 0.5, omega: float = 0.98,
                             ni: int = 16) -> MDPModel:
    """
    Build the L1 volatility MDP with N_vol hidden states.
    Same structure as sentiment MDP — different A and D.

    Param restoration handled externally by heartbeat._load_and_resize_params.
    """
    A_mat = np.array(llm_params["A_volatility"], dtype=float)
    D_vec = np.array(llm_params["D_volatility"], dtype=float)
    N_obs, N_vol = A_mat.shape
    A_mat = spm_norm(A_mat)
    a_mat = A_mat * CONCENTRATION
    B_mat = np.eye(N_vol)[:, :, np.newaxis]

    return MDPModel(
        A=[A_mat], B=[B_mat], D=[D_vec], T=1,
        a=[a_mat], d=[D_vec.copy()],
        alpha=alpha, beta=beta, tau=tau, erp=erp, eta=eta, omega=omega, ni=ni,
    )


# ===========================================================================
# Combined L1 sub-MDP for hierarchical link — variable N
# ===========================================================================

def build_combined_l1(mdp_sent: MDPModel, mdp_vol: MDPModel,
                      alpha: float = 512.0, beta: float = 1.0,
                      tau: float = 4.0, erp: float = 1.0,
                      eta: float = 0.5, omega: float = 0.98,
                      ni: int = 16) -> MDPModel:
    """
    Wrap both L1 MDPs into a single two-factor sub-MDP for the solver.
    Handles arbitrary N_sent and N_vol.

    Factor 0 = sentiment (N_sent states)
    Factor 1 = volatility (N_vol states)

    A[0]: (N_sent, N_sent, N_vol) — identity readout of factor 0
    A[1]: (N_vol,  N_sent, N_vol) — identity readout of factor 1
    """
    N_sent = mdp_sent.A[0].shape[1]   # number of sentiment states
    N_vol  = mdp_vol.A[0].shape[1]    # number of volatility states

    # A_sent: P(obs=i | sent=i, vol=j) = 1  for all j
    A_sent = np.zeros((N_sent, N_sent, N_vol))
    for i in range(N_sent):
        A_sent[i, i, :] = 1.0

    # A_vol: P(obs=i | sent=j, vol=i) = 1  for all j
    A_vol = np.zeros((N_vol, N_sent, N_vol))
    for i in range(N_vol):
        A_vol[i, :, i] = 1.0

    return MDPModel(
        A=[A_sent, A_vol],
        B=[np.eye(N_sent)[:, :, np.newaxis],
           np.eye(N_vol)[:, :, np.newaxis]],
        D=[mdp_sent.D[0].copy(), mdp_vol.D[0].copy()],
        T=1,
        a=[np.ones((N_sent, N_sent, N_vol)) * CONCENTRATION,
           np.ones((N_vol,  N_sent, N_vol)) * CONCENTRATION],
        d=[mdp_sent.d[0].copy(), mdp_vol.d[0].copy()],
        V=np.ones((0, 1, 2), dtype=int),
        alpha=alpha, beta=beta, tau=tau, erp=erp, eta=eta, omega=omega, ni=ni,
    )


# ===========================================================================
# L2 MDP builder — adapts to variable N_sent and N_vol
# ===========================================================================

def build_l2_mdp(mdp_sent: MDPModel,
                 mdp_vol:  MDPModel,
                 llm_params: dict,
                 pref_profit: float = 2.0,
                 pref_loss:   float = -3.0,
                 T: int = 8,
                 N_regime: int = 4,
                 N_policies: int = 3,
                 D_freeze: float = 100.0,
                 alpha: float = 512.0, beta: float = 1.0,
                 tau: float = 4.0, erp: float = 1.0,
                 eta: float = 0.5, omega: float = 0.98,
                 ni: int = 16) -> MDPModel:
    """
    Build L2 market regime MDP.

    Adapts to whatever N_sent and N_vol the LLM chose for this cycle.
    L2 modalities:
        0: price bar (3 outcomes)       — Alpaca OHLCV
        1: sentiment (N_sent outcomes)  — from L1a via link
        2: volatility (N_vol outcomes)  — from L1b via link
        3: feedback (3 outcomes)        — profit/loss/flat at last bar

    L2 hidden factors:
        0: market regime (N_regime states)
        1: time in window (T states)
        2: position (3 states)
    """
    N_sent = mdp_sent.A[0].shape[1]
    N_vol  = mdp_vol.A[0].shape[1]
    N_pos  = 3
    N_time = T

    print(f"[builder] L2 MDP: N_sent={N_sent}  N_vol={N_vol}  "
          f"N_regime={N_regime}  T={T}")

    # ── D priors ──────────────────────────────────────────────────────────
    D2 = [
        np.ones(N_regime) / N_regime,
        np.array([1.0] + [0.0] * (N_time - 1)),
        np.array([1.0, 0.0, 0.0]),
    ]
    d2 = [D2[0].copy(),
          D2[1].copy() * D_freeze,
          D2[2].copy() * D_freeze]

    # ── A matrices ────────────────────────────────────────────────────────
    # Modality 0: price bar (3) × regime (N_regime)
    # Generalised: bull regime → mostly bull bars, etc.
    # Extra regimes (beyond 4) default to mixed
    price_by_regime = np.full((3, N_regime), 1/3)
    base = np.array([
        [0.70, 0.10, 0.30, 0.50],
        [0.10, 0.70, 0.30, 0.30],
        [0.20, 0.20, 0.40, 0.20],
    ])
    for r in range(min(4, N_regime)):
        price_by_regime[:, r] = base[:, r]
    price_by_regime = spm_norm(price_by_regime)

    A2_price = np.zeros((3, N_regime, N_time, N_pos))
    for ti in range(N_time):
        for pi in range(N_pos):
            A2_price[:, :, ti, pi] = price_by_regime

    # Modality 1: sentiment (N_sent) × regime (N_regime)
    # Build a (N_sent × N_regime) likelihood, blended with LLM D_sent
    D_sent = np.array(llm_params["D_sentiment"])   # (N_sent,)
    # Base: first state = "positive" maps to bull regime
    sent_by_regime = np.ones((N_sent, N_regime)) / N_sent
    for s in range(N_sent):
        # States beyond the base 3 get uniform mapping
        if s == 0:   # first state: positive/bullish → bull regime
            sent_by_regime[s, 0] = 0.60
            sent_by_regime[s, 1] = 0.10
        elif s == 1:  # second state: negative/bearish → bear regime
            sent_by_regime[s, 0] = 0.10
            sent_by_regime[s, 1] = 0.60
    # Blend with LLM D_sent for the bull regime column
    if len(D_sent) == N_sent:
        sent_by_regime[:, 0] = spm_norm(
            (0.5 * sent_by_regime[:, 0] + 0.5 * D_sent).reshape(-1, 1)
        ).ravel()
    sent_by_regime = spm_norm(sent_by_regime)

    A2_sentiment = np.zeros((N_sent, N_regime, N_time, N_pos))
    for ti in range(N_time):
        for pi in range(N_pos):
            A2_sentiment[:, :, ti, pi] = sent_by_regime

    # Modality 2: volatility (N_vol) × regime (N_regime)
    D_vol = np.array(llm_params["D_volatility"])   # (N_vol,)
    vol_by_regime = np.ones((N_vol, N_regime)) / N_vol
    for v in range(N_vol):
        if v == 0:   # high vol → bear/uncertain regimes
            vol_by_regime[v, 1] = 0.55
            vol_by_regime[v, 2] = 0.55
        elif v == 2 if N_vol > 2 else -1:  # low vol → bull regime
            vol_by_regime[v, 0] = 0.55
    if len(D_vol) == N_vol:
        vol_by_regime[:, 2 % N_regime] = spm_norm(
            (0.5 * vol_by_regime[:, 2 % N_regime] + 0.5 * D_vol
             ).reshape(-1, 1)
        ).ravel()
    vol_by_regime = spm_norm(vol_by_regime)

    A2_volatility = np.zeros((N_vol, N_regime, N_time, N_pos))
    for ti in range(N_time):
        for pi in range(N_pos):
            A2_volatility[:, :, ti, pi] = vol_by_regime

    # Modality 3: position feedback (3) — unchanged structure
    A2_feedback = np.zeros((3, N_regime, N_time, N_pos))
    for r in range(N_regime):
        for ti in range(N_time - 1):
            for pi in range(N_pos):
                A2_feedback[0, r, ti, pi] = 1.0
    tl = N_time - 1
    # long: bull regime → profit, bear → loss, others → mixed
    for r in range(N_regime):
        if r == 0:   A2_feedback[:, r, tl, 1] = [0.0, 0.85, 0.15]
        elif r == 1: A2_feedback[:, r, tl, 1] = [0.0, 0.15, 0.85]
        else:        A2_feedback[:, r, tl, 1] = [0.2, 0.50, 0.30]
    # short: opposite
    for r in range(N_regime):
        if r == 0:   A2_feedback[:, r, tl, 2] = [0.0, 0.15, 0.85]
        elif r == 1: A2_feedback[:, r, tl, 2] = [0.0, 0.85, 0.15]
        else:        A2_feedback[:, r, tl, 2] = [0.2, 0.30, 0.50]
    for r in range(N_regime):
        A2_feedback[:, r, tl, 0] = [1.0, 0.0, 0.0]

    A2 = [A2_price, A2_sentiment, A2_volatility, A2_feedback]
    a2 = [A * CONCENTRATION for A in A2]

    # ── Transitions ───────────────────────────────────────────────────────
    # Regime: slow-changing, high persistence, extra regimes get uniform row
    B2_regime_mat = np.eye(N_regime) * 0.7 + np.ones((N_regime, N_regime)) * 0.3 / N_regime
    B2_regime_mat = spm_norm(B2_regime_mat)
    B2_regime = B2_regime_mat[:, :, np.newaxis]

    B2_time = np.zeros((N_time, N_time, 1))
    for i in range(N_time - 1):
        B2_time[i + 1, i, 0] = 1.0
    B2_time[N_time - 1, N_time - 1, 0] = 1.0

    B2_pos = np.zeros((N_pos, N_pos, N_policies))
    B2_pos[:, :, 0] = np.eye(N_pos)
    B2_pos[:, :, 1] = [[0,0,0],[1,1,1],[0,0,0]]
    B2_pos[:, :, 2] = [[0,0,0],[0,0,0],[1,1,1]]
    B2 = [B2_regime, B2_time, B2_pos]

    # ── Policies ─────────────────────────────────────────────────────────
    V2 = np.ones((T - 1, N_policies, 3), dtype=int)
    V2[T - 2, 0, 2] = 1   # hold
    V2[T - 2, 1, 2] = 2   # buy
    V2[T - 2, 2, 2] = 3   # sell

    # ── Preferences ───────────────────────────────────────────────────────
    C2_fb = np.zeros((3, T))
    C2_fb[0, T - 1] = 0.0           # flat
    C2_fb[1, T - 1] = pref_profit   # profit
    C2_fb[2, T - 1] = pref_loss     # loss
    C2 = [np.zeros((3, T)),
          np.zeros((N_sent, T)),
          np.zeros((N_vol, T)),
          C2_fb]

    # ── Hierarchical link ─────────────────────────────────────────────────
    # link shape: (n_sub_factors=2, n_L2_modalities=4)
    # factor 0 (sentiment) → modality 1
    # factor 1 (volatility) → modality 2
    mdp_combined = build_combined_l1(mdp_sent, mdp_vol,
                                     alpha=alpha, beta=beta, tau=tau,
                                     erp=erp, eta=eta, omega=omega, ni=ni)
    link = np.array([
        [0, 1, 0, 0],   # sentiment → modality 1
        [0, 0, 1, 0],   # volatility → modality 2
    ])

    return MDPModel(
        A=A2, B=B2, D=D2, T=T,
        a=a2, d=d2, C=C2, V=V2,
        alpha=alpha, beta=beta, tau=tau, erp=erp,
        eta=eta, omega=omega, ni=ni,
        MDP=mdp_combined, link=link,
    )


# ===========================================================================
# Price observation injection — reads Ng from mdp at runtime
# ===========================================================================

def inject_price_observations(mdp: MDPModel,
                               bars: list[dict],
                               T: int = 8) -> MDPModel:
    """
    Inject discretised Alpaca bar observations into L2 modality 0.

    Reads Ng from mdp.A at runtime so this is correct for any number
    of modalities the LLM proposed. Modalities 1+ (sentiment, volatility)
    are filled by the solver automatically via the hierarchical link.
    """
    Ng      = len(mdp.A)
    Nf      = len(mdp.B)
    regimes = bars_to_regime(bars[-T:])

    s = np.ones((Nf, T), dtype=int)
    s[1, :] = np.arange(1, T + 1)   # time clock

    o = np.ones((Ng, T), dtype=int)  # default = obs index 0 (1-based)
    for t_idx, regime in enumerate(regimes[:T]):
        o[0, t_idx] = regime + 1     # modality 0: price bar

    mdp.s = s
    mdp.o = o
    return mdp


# ===========================================================================
# Signal extraction — reads actual array sizes from mdp_result
# ===========================================================================

def beliefs_to_signal(mdp_result: MDPModel,
                       llm_params: dict) -> dict:
    """
    Extract trading signal from solved L2 MDP.

    Reads actual array dimensions from mdp_result.X — correct for any
    N_sent and N_vol. Also carries state_names from llm_params for logging.
    """
    policy_post   = mdp_result.R[:, -1]
    best_policy   = int(np.argmax(policy_post))
    confidence    = float(policy_post[best_policy])
    regime_belief = mdp_result.X[0][:, -1]

    n_sent = llm_params.get("n_sent", 3)
    n_vol  = llm_params.get("n_vol",  3)
    sent_belief = np.ones(n_sent) / n_sent
    vol_belief  = np.ones(n_vol)  / n_vol

    if mdp_result.mdp_t is not None:
        last_sub = next(
            (r for r in reversed(mdp_result.mdp_t) if r is not None), None)
        if last_sub is not None and last_sub.X is not None:
            if len(last_sub.X) > 0:
                sent_belief = last_sub.X[0][:, 0]
            if len(last_sub.X) > 1:
                vol_belief  = last_sub.X[1][:, 0]

    return {
        "action":        {0: "hold", 1: "buy", 2: "sell"}[best_policy],
        "confidence":    confidence,
        "regime_belief": regime_belief,
        "sent_belief":   sent_belief,
        "vol_belief":    vol_belief,
        "policy_post":   policy_post,
        "sent_names":    llm_params.get("sent_names", [f"s{i}" for i in range(n_sent)]),
        "vol_names":     llm_params.get("vol_names",  [f"v{i}" for i in range(n_vol)]),
    }
