"""
test_active_inference.py
========================
Validation test suite for the active inference hierarchical POMDP model.
Matches Step_by_Step_Hierarchical_Model.m (Smith, Friston, Whyte).

Run with:
    python3 test_active_inference.py           # all tiers
    python3 test_active_inference.py --tier 1  # specific tier only
    python3 test_active_inference.py --fast    # skip slow trial simulations

Each tier assumes the previous tiers pass.
"""

import sys
import argparse
import traceback
import numpy as np
import copy

# ── helpers ──────────────────────────────────────────────────────────────────

PASS  = "\033[32m  PASS\033[0m"
FAIL  = "\033[31m  FAIL\033[0m"
SKIP  = "\033[33m  SKIP\033[0m"
BOLD  = "\033[1m"
RESET = "\033[0m"

results = {"pass": 0, "fail": 0, "skip": 0}


def check(name, expr, detail=""):
    """Assert expr is truthy; print result line."""
    try:
        ok = bool(expr)
    except Exception as e:
        ok = False
        detail = str(e)
    if ok:
        print(f"{PASS}  {name}")
        results["pass"] += 1
    else:
        print(f"{FAIL}  {name}" + (f"\n         {detail}" if detail else ""))
        results["fail"] += 1


def section(title):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


def skip(name, reason=""):
    print(f"{SKIP}  {name}" + (f"  ({reason})" if reason else ""))
    results["skip"] += 1


# ── build the model (reused across tiers) ────────────────────────────────────

def build_level1(pr1, concentration_scale, alpha, beta, tau, erp, eta, omega):
    from .utils     import spm_softmax
    from .mdp_model import MDPModel
    D1 = [np.array([1.0, 1.0])]
    A1 = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    a1 = [spm_softmax(pr1 * np.log(A1[0] + np.exp(-4.0))) * concentration_scale]
    B1 = [np.eye(2)[:, :, np.newaxis]]
    return MDPModel(A=A1, B=B1, D=D1, T=1, a=a1, d=[D1[0].copy()],
                    alpha=alpha, beta=beta, tau=tau, erp=erp, eta=eta, omega=omega)


def build_level2(T, Pi, pr2, concentration_scale, alpha, beta, tau, erp,
                 eta, omega, d_freeze, pref_correct, pref_incorrect, MDP_1):
    from .utils     import spm_softmax
    from .mdp_model import MDPModel

    Nf = 3
    D2 = [np.array([1., 1., 1., 1.]),
          np.array([1., 0., 0., 0., 0., 0.]),
          np.array([1., 0., 0.])]
    d2 = [D2[0].copy(), D2[1].copy() * d_freeze, D2[2].copy() * d_freeze]

    A2_tone = np.zeros((2, 4, 6, 3))
    for i in range(6):
        for j in range(3):
            A2_tone[:, :, i, j] = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    for j in range(3):
        A2_tone[:, :, 3, j] = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])

    A2_report = np.zeros((3, 4, 6, 3))
    for i in range(6):
        for j in range(3):
            A2_report[:, :, i, j] = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    A2_report[:, :, 5, 1] = np.array([[0,0,0,0],[0,0,1,1],[1,1,0,0]])
    A2_report[:, :, 5, 2] = np.array([[0,0,0,0],[1,1,0,0],[0,0,1,1]])

    A2 = [A2_tone, A2_report]
    a2 = [spm_softmax(pr2 * np.log(A2_tone + np.exp(-4.0))) * concentration_scale,
          A2_report.copy() * concentration_scale]

    B2_seq  = np.eye(4)[:, :, np.newaxis]
    B2_time = np.zeros((6, 6, 1))
    for i in range(5):
        B2_time[i + 1, i, 0] = 1.0
    B2_time[5, 5, 0] = 1.0

    B2_report = np.zeros((3, 3, 3))
    B2_report[:, :, 0] = np.array([[1,1,1],[0,0,0],[0,0,0]])
    B2_report[:, :, 1] = np.array([[0,0,0],[1,1,1],[0,0,0]])
    B2_report[:, :, 2] = np.array([[0,0,0],[0,0,0],[1,1,1]])
    B2 = [B2_seq, B2_time, B2_report]

    V2          = np.ones((T - 1, Pi, Nf), dtype=int)
    V2[4, 0, 2] = 2
    V2[4, 1, 2] = 3

    C2_tone   = np.zeros((2, T))
    C2_report = np.zeros((3, T))
    C2_report[1, 5] = pref_incorrect
    C2_report[2, 5] = pref_correct

    return MDPModel(
        A=A2, B=B2, D=D2, T=T, a=a2, d=d2, C=[C2_tone, C2_report], V=V2,
        alpha=alpha, beta=beta, tau=tau, erp=erp, eta=eta, omega=omega,
        MDP=MDP_1, link=np.array([[1, 0]]))


# ═══════════════════════════════════════════════════════════════════════════
#  TIER 1 — Math utilities
# ═══════════════════════════════════════════════════════════════════════════

def run_tier1():
    section("Tier 1 — Math utilities (utils.py)")
    from .utils import spm_softmax, spm_norm, spm_wnorm, spm_KL_dir, spm_log

    # spm_softmax
    x = np.array([1.0, 2.0, 3.0])
    s = spm_softmax(x)
    check("spm_softmax sums to 1",
          abs(s.sum() - 1.0) < 1e-10)
    check("spm_softmax all values positive",
          np.all(s > 0))
    check("spm_softmax largest input gets largest output",
          s[2] > s[1] > s[0])
    check("spm_softmax is global (not per-axis) for 2-D input",
          abs(spm_softmax(np.array([[1., 2.], [3., 4.]])).sum() - 1.0) < 1e-10)

    # spm_norm
    A = np.array([[3.0, 1.0], [2.0, 4.0]])
    N = spm_norm(A)
    check("spm_norm columns sum to 1",
          np.allclose(N.sum(axis=0), 1.0))
    check("spm_norm handles zero column without crash (returns finite values)",
          np.all(np.isfinite(spm_norm(np.array([[0., 1.], [0., 1.]])))))

    # spm_wnorm — equation 40: W = ½(1/sum - 1/a)
    W = spm_wnorm(A)
    col_sums = A.sum(axis=0, keepdims=True)
    expected = (1.0 / col_sums - 1.0 / A) / 2.0
    check("spm_wnorm matches equation 40: ½(1/sum − 1/a)",
          np.allclose(W, expected))
    check("spm_wnorm includes the ½ factor (not just 1/sum - 1/a)",
          np.allclose(W * 2, (1.0 / col_sums - 1.0 / A)))

    # spm_KL_dir
    p = np.array([50.0, 30.0, 20.0])
    check("spm_KL_dir(p, p) = 0",
          abs(spm_KL_dir(p, p)) < 1e-8)
    check("spm_KL_dir ≥ 0",
          spm_KL_dir(np.array([60., 20., 20.]), p) >= 0.0)
    check("spm_KL_dir no nan with zero entries",
          np.isfinite(spm_KL_dir(np.array([50., 0., 50.]), np.array([50., 0., 50.]))))
    check("spm_KL_dir increases with greater divergence",
          spm_KL_dir(np.array([80., 10., 10.]), p) >
          spm_KL_dir(np.array([55., 25., 20.]), p))

    # spm_log
    check("spm_log never returns -inf (floors at 1e-16)",
          np.all(np.isfinite(spm_log(np.array([0.0, 1e-300, 1.0])))))
    check("spm_log(1) = 0",
          abs(spm_log(np.array([1.0]))[0]) < 1e-10)
    check("spm_log(e) ≈ 1",
          abs(spm_log(np.array([np.e]))[0] - 1.0) < 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
#  TIER 2 — Single-level MDP (Level-1 only, no hierarchy)
# ═══════════════════════════════════════════════════════════════════════════

def run_tier2():
    section("Tier 2 — Single-level MDP (no hierarchy)")
    from .utils     import spm_softmax
    from .mdp_model import MDPModel
    from .solver    import spm_MDP_VB_X
    from .config    import NI as Ni, TAU, ALPHA, BETA, ERP, ETA, OMEGA, PR1, CONCENTRATION_SCALE

    MDP_1 = build_level1(PR1, CONCENTRATION_SCALE, ALPHA, BETA, TAU, ERP, ETA, OMEGA)

    # --- A: belief normalisation ---
    np.random.seed(0)
    r = spm_MDP_VB_X(copy.deepcopy(MDP_1))

    for f in range(len(r.xn)):
        xn_f = r.xn[f]   # (Ni, Ns, S, T, Np)
        sums = xn_f.sum(axis=1)   # sum over states
        check(f"xn[{f}] beliefs sum to 1 at every (Ni, S, T, Np)",
              np.allclose(sums, 1.0, atol=1e-6))

    check("X (model-averaged beliefs) sums to 1 per timestep",
          all(np.allclose(r.X[f].sum(axis=0), 1.0, atol=1e-6)
              for f in range(len(r.X))))

    # --- B: deterministic A gives correct inference ---
    # Force state 1 (LOW) via 1-based index 2. State 0 cannot be forced this way
    # because s=1 (1-based) maps to s[f,0]=0, which is the solver's "not set" sentinel.
    # s=2 (1-based) → 0-based index 1 = LOW. Identity A → P(LOW) ≈ 1.
    mdp_det = copy.deepcopy(MDP_1)
    mdp_det.s = np.array([2])   # force state 1/LOW (1-based)
    r_det = spm_MDP_VB_X(mdp_det)
    check("Deterministic A: posterior concentrates on observed state (LOW)",
          r_det.X[0][1, 0] > 0.9,
          f"P(state=LOW) = {r_det.X[0][1,0]:.3f}")

    # --- C: free energy: F for the chosen policy should be more negative ---
    check("F values are finite",
          np.all(np.isfinite(r.F)))
    check("Policy posterior R sums to 1",
          np.allclose(r.R.sum(axis=0), 1.0, atol=1e-6))

    # --- D: learning updates concentration parameters ---
    a_before = MDP_1.a[0].copy()
    r_learn = spm_MDP_VB_X(copy.deepcopy(MDP_1))
    a_after = r_learn.a[0]
    check("Learning increases total concentration mass (eta=1, omega=1)",
          a_after.sum() > a_before.sum(),
          f"before={a_before.sum():.1f}  after={a_after.sum():.1f}")

    # --- E: with high-precision A, inference is fast ---
    # Run again and check beliefs converge (xn[f][-1] ≈ xn[f][-2])
    xn_f = r.xn[0]  # (Ni, Ns, S, T, Np)
    last  = xn_f[-1, :, 0, 0, :]
    prev  = xn_f[-2, :, 0, 0, :]
    check("Beliefs converge: last two VB iterations are close",
          np.allclose(last, prev, atol=1e-4),
          f"max diff = {np.abs(last - prev).max():.2e}")


# ═══════════════════════════════════════════════════════════════════════════
#  TIER 3 — Hierarchical link
# ═══════════════════════════════════════════════════════════════════════════

def run_tier3():
    section("Tier 3 — Hierarchical link")
    from .solver import spm_MDP_VB_X
    from .config import (NI as Ni, TAU, ALPHA, BETA, ERP, ETA, OMEGA,
                        PR1, PR2, CONCENTRATION_SCALE, T, N_POLICIES as Pi,
                        PREF_CORRECT, PREF_INCORRECT, D_FREEZE_SCALE)

    MDP_1 = build_level1(PR1, CONCENTRATION_SCALE, ALPHA, BETA, TAU, ERP, ETA, OMEGA)
    mdp   = build_level2(T, Pi, PR2, CONCENTRATION_SCALE, ALPHA, BETA, TAU, ERP,
                         ETA, OMEGA, D_FREEZE_SCALE, PREF_CORRECT, PREF_INCORRECT, MDP_1)

    np.random.seed(42)
    mdp_ldgs = copy.deepcopy(mdp); mdp_ldgs.s = np.array([3])
    r_ldgs   = spm_MDP_VB_X(mdp_ldgs)

    np.random.seed(42)
    mdp_lsgd = copy.deepcopy(mdp); mdp_lsgd.s = np.array([3])
    r_lsgd   = spm_MDP_VB_X(mdp_lsgd)

    # --- A: sub-MDPs were called ---
    check("Sub-MDPs called for every outer timestep",
          r_ldgs.mdp_t is not None and all(s is not None for s in r_ldgs.mdp_t))

    # --- B: L2 outcome is soft, not one-hot ---
    # Access the O vector stored in the solved result via sub-MDP soft output
    sub0 = r_ldgs.mdp_t[0]
    check("Sub-MDP result has xn (beliefs were stored)",
          sub0 is not None and sub0.xn is not None)

    # Verify L2 outcomes are soft: X[0][:,0] of sub-MDP should not be one-hot
    sub_X = sub0.X[0][:, 0]
    check("L2 outcome from sub-MDP is soft (not one-hot): max < 0.9999",
          sub_X.max() < 0.9999,
          f"max value = {sub_X.max():.4f}")
    check("L2 outcome from sub-MDP sums to 1",
          abs(sub_X.sum() - 1.0) < 1e-6,
          f"sum = {sub_X.sum():.6f}")

    # --- C: sub-MDP prior D comes from L2 beliefs ---
    # D should differ between trials because L2 beliefs differ
    # Compare sub-MDP at t=0 for LDGS vs a fresh uninformed run
    sub_D = sub0.D[0] if sub0.D is not None else None
    check("Sub-MDP prior D is set (not None)",
          sub_D is not None)
    check("Sub-MDP prior D is normalised",
          sub_D is not None and abs(sub_D.sum() - 1.0) < 1e-6)

    # --- D: condition contrast — within LDGS trial, t=0 (standard) vs t=3 (oddball) ---
    # On the first trial, LDGS and LSGD have the same s=[3] prior, so their
    # sub-MDP D priors are identical. The meaningful contrast is within one LDGS
    # trial: the sub-MDP D at the oddball timestep t=3 differs from t=0 because
    # L2 beliefs evolve across the trial as evidence accumulates.
    sub_t0 = r_ldgs.mdp_t[0]
    sub_t3 = r_ldgs.mdp_t[3]

    xn_t0 = sub_t0.xn[0][:, :, 0, 0, :].mean(axis=-1)   # (Ni, Ns)
    xn_t3 = sub_t3.xn[0][:, :, 0, 0, :].mean(axis=-1)

    low_t0_end = xn_t0[-1, 1]   # P(LOW) at end of t=0 VB
    low_t3_end = xn_t3[-1, 1]   # P(LOW) at end of t=3 VB (oddball)

    check("Within LDGS: P(LOW) at oddball t=3 >> P(LOW) at standard t=0",
          low_t3_end > low_t0_end + 0.3,
          f"P(LOW) t=0={low_t0_end:.3f}  P(LOW) t=3={low_t3_end:.3f}")

    # State 1 (LOW) trajectory at oddball: LDGS should converge to high P(LOW)
    check("LDGS sub-MDP at t=3: P(LOW) converges high (heard LOW tone)",
          low_t3_end > 0.8,
          f"P(LOW) at end = {low_t3_end:.3f}")

    # --- E: tones at non-oddball timesteps differ from oddball ---
    xn_t0 = r_ldgs.mdp_t[0].xn[0][:, :, 0, 0, :].mean(axis=-1)
    xn_t3 = r_ldgs.mdp_t[3].xn[0][:, :, 0, 0, :].mean(axis=-1)
    check("LDGS: L1 xn at t=0 (standard) differs from t=3 (oddball)",
          abs(xn_t0[-1, 0] - xn_t3[-1, 0]) > 0.3,
          f"t=0 P(HIGH)={xn_t0[-1,0]:.3f}  t=3 P(HIGH)={xn_t3[-1,0]:.3f}")

    # --- F: L2 beliefs are valid ---
    check("L2 X[0] (sequence beliefs) sums to 1 at each timestep",
          np.allclose(r_ldgs.X[0].sum(axis=0), 1.0, atol=1e-6))
    check("L2 R (policy posterior) sums to 1",
          np.allclose(r_ldgs.R.sum(axis=0), 1.0, atol=1e-6))


# ═══════════════════════════════════════════════════════════════════════════
#  TIER 4 — Trial-by-trial learning
# ═══════════════════════════════════════════════════════════════════════════

def run_tier4():
    section("Tier 4 — Trial-by-trial learning (10 trials)")
    from .solver     import spm_MDP_VB_X
    from .run_trials import run_trials
    from .config     import (NI as Ni, TAU, ALPHA, BETA, ERP, ETA, OMEGA,
                            PR1, PR2, CONCENTRATION_SCALE, T, N_POLICIES as Pi,
                            PREF_CORRECT, PREF_INCORRECT, D_FREEZE_SCALE, N_TRIALS as N)

    MDP_1 = build_level1(PR1, CONCENTRATION_SCALE, ALPHA, BETA, TAU, ERP, ETA, OMEGA)
    mdp   = build_level2(T, Pi, PR2, CONCENTRATION_SCALE, ALPHA, BETA, TAU, ERP,
                         ETA, OMEGA, D_FREEZE_SCALE, PREF_CORRECT, PREF_INCORRECT, MDP_1)

    np.random.seed(42)
    mdp_c1 = copy.deepcopy(mdp); mdp_c1.s = np.array([3])
    MDP_LDGS = run_trials(mdp_c1, N)

    # --- A: concentration params update across trials ---
    # With omega=1 (MATLAB default), the forgetting rule is:
    #   a_new = a_0 + da * eta   (resets to baseline each trial)
    # Total mass does NOT accumulate — it stays ≈ a_0.sum() + eta per trial.
    # What changes is the DISTRIBUTION (which entries are larger/smaller),
    # encoding which outcomes were observed. Check normalised a instead.
    from .utils import spm_norm as _spm_norm
    A_norm_t1  = _spm_norm(MDP_LDGS[0].a[0].reshape(2, -1))
    A_norm_t10 = _spm_norm(MDP_LDGS[-1].a[0].reshape(2, -1))
    check("Normalised a[0] distribution shifts across 10 trials (omega=1: correct)",
          np.abs(A_norm_t1 - A_norm_t10).max() > 1e-6,
          f"max diff={np.abs(A_norm_t1 - A_norm_t10).max():.2e}")

    # --- B: params propagated correctly between trials ---
    check("Trial 2 a ≠ trial 1 a (learning actually applied)",
          not np.allclose(MDP_LDGS[0].a[0], MDP_LDGS[1].a[0]))
    check("Trial 10 a ≠ trial 1 a (10 trials of updates)",
          not np.allclose(MDP_LDGS[0].a[0], MDP_LDGS[-1].a[0]))

    # --- C: after 10 LDGS trials, agent confident about HIGH-LOW ---
    X_seq_t10 = MDP_LDGS[-1].X[0]   # (4, T)
    state2_belief = X_seq_t10[2, 0]  # HIGH-LOW sequence at t=0
    check("After 10 LDGS trials: P(HIGH-LOW sequence) > 0.5 at t=0",
          state2_belief > 0.5,
          f"P(HIGH-LOW) = {state2_belief:.3f}")

    # --- D: sub-level learning propagates ---
    # L1 uses omega=1 too, so its a also resets to a_0 + da each trial.
    # Check that the L1 normalised a differs between trials, confirming
    # the sub-MDP's learning parameters are actually being updated.
    sub_t1  = MDP_LDGS[0].mdp_t[0]
    sub_t10 = MDP_LDGS[-1].mdp_t[0]
    if sub_t1 is not None and sub_t10 is not None and sub_t1.a and sub_t10.a:
        from .utils import spm_norm as _spm_norm
        sub_a_norm_t1  = _spm_norm(sub_t1.a[0])
        sub_a_norm_t10 = _spm_norm(sub_t10.a[0])
        check("L1 sub-level normalised a[0] differs between trial 1 and 10",
              np.abs(sub_a_norm_t1 - sub_a_norm_t10).max() > 1e-6,
              f"max diff={np.abs(sub_a_norm_t1 - sub_a_norm_t10).max():.2e}")
    else:
        skip("L1 sub-level learning check", "sub-MDP a not available")

    # --- E: LSGD trial 10 (global deviant) surprises the agent ---
    np.random.seed(99)
    MDP_LSGD = []
    mdp_c2 = copy.deepcopy(mdp); mdp_c2.s = np.array([3])
    for i in range(N):
        trial = copy.deepcopy(mdp_c2)
        if i == N - 1: trial.s = np.array([1])
        if i > 0:
            prev = MDP_LSGD[i - 1]
            if prev.a: trial.a = [p.copy() for p in prev.a]; trial.a_0 = [p.copy() for p in prev.a_0]
            if prev.d: trial.d = [p.copy() for p in prev.d]; trial.d_0 = [p.copy() for p in prev.d_0]
        MDP_LSGD.append(spm_MDP_VB_X(trial))

    # On trial 10, LSGD sequence is different → L2 seq-type posterior differs
    seq_ldgs_t10 = MDP_LDGS[-1].X[0][:, 0]
    seq_lsgd_t10 = MDP_LSGD[-1].X[0][:, 0]
    check("Trial 10: L2 sequence beliefs differ between LDGS and LSGD",
          np.abs(seq_ldgs_t10 - seq_lsgd_t10).max() > 0.1,
          f"max diff = {np.abs(seq_ldgs_t10 - seq_lsgd_t10).max():.3f}")

    return MDP_LDGS, MDP_LSGD


# ═══════════════════════════════════════════════════════════════════════════
#  TIER 5 — ERP shape validation
# ═══════════════════════════════════════════════════════════════════════════

def run_tier5(MDP_LDGS=None, MDP_LSGD=None):
    section("Tier 5 — ERP shape validation")
    from .config import NI as Ni, T, N_TRIALS as N

    if MDP_LDGS is None or MDP_LSGD is None:
        skip("All Tier 5 checks", "requires Tier 4 trial data (run without --fast)")
        return

    ldgs = MDP_LDGS[-1]
    lsgd = MDP_LSGD[-1]

    # ── ERP extraction (mirrors Step_by_Step_Hierarchical_Model.py) ──────

    def build_L1(res, state_idx=1):
        segs = []
        for t in range(T):
            sub = res.mdp_t[t] if res.mdp_t else None
            seg = np.zeros(Ni)
            if sub and sub.xn:
                xn_f = sub.xn[0]
                R    = sub.R if sub.R is not None else np.ones((1, 1))
                for k in range(xn_f.shape[-1]):
                    seg += xn_f[:, state_idx, 0, 0, k] * R[k, 0]
            segs.append(seg)
        return np.gradient(np.concatenate(segs))

    def build_L2(res, state_idx=2):
        R    = res.R
        xn_f = res.xn[0]
        Ni_, Ns_, S_, T_, Np_ = xn_f.shape
        sig  = np.zeros(Ni_ * T_)
        for t in range(T_):
            base = t * Ni_
            if S_ > t:
                for k in range(Np_):
                    sig[base:base + Ni_] += xn_f[:, state_idx, t, t, k] * R[k, t]
        return np.gradient(sig)

    from scipy.ndimage import uniform_filter1d
    sm = lambda x: uniform_filter1d(x.astype(float), size=5)

    v_ldgs = sm(build_L1(ldgs))   # LDGS Level-1 (local deviant)
    v_lsgd = sm(build_L1(lsgd))   # LSGD Level-1 (local standard)
    u_ldgs = sm(build_L2(ldgs))   # LDGS Level-2 (global standard)
    u_lsgd = sm(build_L2(lsgd))   # LSGD Level-2 (global deviant)

    L1_lo, L1_hi = 69, 120
    L2_lo, L2_hi = 95, 140

    # --- A: signals are above noise ---
    check("L1 signal is non-trivial (> 1e-6, not floating-point noise)",
          v_ldgs[L1_lo:L1_hi].max() > 1e-6,
          f"max = {v_ldgs[L1_lo:L1_hi].max():.2e}")
    check("L2 signal is non-trivial (> 1e-6)",
          u_ldgs[L2_lo:L2_hi].max() > 1e-6,
          f"max = {u_ldgs[L2_lo:L2_hi].max():.2e}")

    # --- B: MMN shape ---
    check("L1: deviant (LDGS) peaks higher than standard (LSGD) — paper: red > blue",
          v_ldgs[L1_lo:L1_hi].max() > v_lsgd[L1_lo:L1_hi].max(),
          f"LDGS max={v_ldgs[L1_lo:L1_hi].max():.4f}  "
          f"LSGD max={v_lsgd[L1_lo:L1_hi].max():.4f}")

    mmn = v_lsgd[L1_lo:L1_hi] - v_ldgs[L1_lo:L1_hi]
    check("MMN (standard − deviant) has a negative dip — paper: dips below 0",
          mmn.min() < -0.005,
          f"MMN min = {mmn.min():.4f}")

    # --- C: P300 shape ---
    check("L2: global standard (LDGS) peaks higher than global deviant (LSGD)",
          u_ldgs[L2_lo:L2_hi].max() > u_lsgd[L2_lo:L2_hi].max(),
          f"LDGS max={u_ldgs[L2_lo:L2_hi].max():.4f}  "
          f"LSGD max={u_lsgd[L2_lo:L2_hi].max():.4f}")

    p300 = u_ldgs[L2_lo:L2_hi] - u_lsgd[L2_lo:L2_hi]
    check("P300 (global standard − global deviant) has a positive peak",
          p300.max() > 0.005,
          f"P300 max = {p300.max():.4f}")

    # --- D: signals differ between conditions ---
    check("L1 conditions differ by > 0.01 somewhere in the window",
          np.abs(v_ldgs[L1_lo:L1_hi] - v_lsgd[L1_lo:L1_hi]).max() > 0.01)
    check("L2 conditions differ by > 0.01 somewhere in the window",
          np.abs(u_ldgs[L2_lo:L2_hi] - u_lsgd[L2_lo:L2_hi]).max() > 0.01)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Active inference test suite")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only this tier")
    parser.add_argument("--fast", action="store_true",
                        help="Skip slow multi-trial simulations (Tiers 4 & 5)")
    args = parser.parse_args()

    # Add current directory to path so imports work
    sys.path.insert(0, ".")

    print(f"\n{BOLD}Active inference validation suite{RESET}")
    print("Comparing against Step_by_Step_Hierarchical_Model.m")

    run_all = args.tier is None
    trial_data = None   # carry Tier 4 output into Tier 5

    try:
        if run_all or args.tier == 1:
            run_tier1()

        if run_all or args.tier == 2:
            run_tier2()

        if run_all or args.tier == 3:
            run_tier3()

        if run_all or args.tier == 4:
            if args.fast:
                section("Tier 4 — Trial-by-trial learning")
                skip("All Tier 4 checks", "--fast mode")
            else:
                trial_data = run_tier4()

        if run_all or args.tier == 5:
            if args.fast:
                section("Tier 5 — ERP shape validation")
                skip("All Tier 5 checks", "--fast mode")
            else:
                ldgs, lsgd = trial_data if trial_data else (None, None)
                run_tier5(ldgs, lsgd)

    except Exception:
        print(f"\n{FAIL}  Unhandled exception — stack trace:")
        traceback.print_exc()
        results["fail"] += 1

    # Summary
    total = results["pass"] + results["fail"] + results["skip"]
    print(f"\n{'═'*60}")
    print(f"  {BOLD}Results:{RESET}  "
          f"\033[32m{results['pass']} passed\033[0m  "
          f"\033[31m{results['fail']} failed\033[0m  "
          f"\033[33m{results['skip']} skipped\033[0m  "
          f"(of {total} checks)")
    print(f"{'═'*60}\n")

    sys.exit(0 if results["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
