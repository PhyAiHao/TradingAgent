"""
learning.py  —  Dirichlet parameter updates (fixed)

Mismatch fix:
  learning.py referenced mdp.C_0 which does not exist in MDPModel.
  The C update block was using uppercase C (fixed preferences) as if
  it were a learnable lowercase c parameter. In the MATLAB code, only
  a lowercase mdp.c field (separate from mdp.C) is learnable.
  Since this model uses fixed preferences (uppercase C only), the C
  update block is removed. All other updates (a, b, d, e) are correct.
"""
import numpy as np
from .utils import spm_KL_dir


def update_parameters(mdp, O, X, u, V, T, Nf, Ng, Np):
    """
    Accumulate Dirichlet concentration parameters after one trial.

    O : O[g][t]  — (No[g],) outcome likelihood vector
    X : X[f]     — (Ns[f], T) Bayesian-model-averaged beliefs
    u : (Np, T)  — policy posteriors
    V : (T-1, Np, Nf) — policy array (1-based actions)
    """
    eta   = mdp.eta
    omega = mdp.omega

    # ------------------------------------------------------------------
    # Update A (likelihood mapping)
    # ------------------------------------------------------------------
    if mdp.a is not None:
        for t in range(T):
            for g in range(Ng):
                # Build N-D sufficient statistic via np.multiply.outer:
                # da shape = (No[g], Ns[0], Ns[1], ..., Ns[Nf-1])
                # matches mdp.a[g].shape exactly — no reshape needed.
                da = np.array(O[g][t], dtype=float)
                for f in range(Nf):
                    da = np.multiply.outer(da, X[f][:, t])
                mask     = mdp.a[g] > 0
                da       = da * mask
                mdp.a[g] = ((mdp.a[g] - mdp.a_0[g]) * (1.0 - omega)
                            + mdp.a_0[g] + da * eta)

    # ------------------------------------------------------------------
    # Update B (transition probabilities)
    # ------------------------------------------------------------------
    if mdp.b is not None:
        for t in range(1, T):
            for f in range(Nf):
                for k in range(Np):
                    v    = int(V[t - 1, k, f]) - 1   # 0-based action
                    db   = u[k, t] * np.outer(X[f][:, t], X[f][:, t - 1])
                    mask = mdp.b[f][:, :, v] > 0
                    db   = db * mask
                    mdp.b[f][:, :, v] = (
                        (mdp.b[f][:, :, v] - mdp.b_0[f][:, :, v]) * (1.0 - omega)
                        + mdp.b_0[f][:, :, v] + db * eta
                    )

    # ------------------------------------------------------------------
    # Update C (preferences) — REMOVED
    # ------------------------------------------------------------------
    # MDPModel stores fixed preferences in uppercase C with no baseline
    # copy C_0.  In MATLAB, only a separate lowercase mdp.c field is
    # learnable.  Since this model uses fixed C, no update is performed.
    # If you add a learnable 'c' field to MDPModel in the future, add
    # its update here, guarded by:  if mdp.c is not None:

    # ------------------------------------------------------------------
    # Update D (initial state priors)
    # ------------------------------------------------------------------
    if mdp.d is not None:
        for f in range(Nf):
            i = mdp.d[f] > 0
            mdp.d[f][i] = (
                (mdp.d[f][i] - mdp.d_0[f][i]) * (1.0 - omega)
                + mdp.d_0[f][i] + X[f][i, 0] * eta
            )

    # ------------------------------------------------------------------
    # Update E (policy priors)
    # ------------------------------------------------------------------
    if mdp.e is not None:
        mdp.e = (
            (mdp.e - mdp.e_0) * (1.0 - omega)
            + mdp.e_0 + eta * u[:, T - 1]
        )

    # ------------------------------------------------------------------
    # Free energy of parameters (KL complexity costs)
    # ------------------------------------------------------------------
    Fa, Fb, Fd = {}, {}, {}

    for g in range(Ng):
        if mdp.a is not None:
            try:
                Fa[g] = -spm_KL_dir(mdp.a[g].ravel(), mdp.a_0[g].ravel())
            except Exception:
                Fa[g] = 0.0

    for f in range(Nf):
        if mdp.b is not None:
            try:
                Fb[f] = -spm_KL_dir(mdp.b[f].ravel(), mdp.b_0[f].ravel())
            except Exception:
                Fb[f] = 0.0
        if mdp.d is not None:
            try:
                Fd[f] = -spm_KL_dir(mdp.d[f].ravel(), mdp.d_0[f].ravel())
            except Exception:
                Fd[f] = 0.0

    return mdp, Fa, Fb, Fd
