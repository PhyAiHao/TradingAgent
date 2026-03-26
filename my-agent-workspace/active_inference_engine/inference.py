"""
inference.py  —  Variational free energy minimisation
General-purpose active inference VB engine.
No model-specific parameters — all hyperparameters are passed as arguments.
"""
import numpy as np
import string
from .utils import spm_log, spm_softmax, spm_wnorm

_LETTERS = string.ascii_lowercase


# ---------------------------------------------------------------------------
# Tensor contraction helpers
# ---------------------------------------------------------------------------

def _marginalise_L_except_f(L, xq, f_skip, Nf):
    """Contract L (Ns0x...xNsNf-1) over all factors except f_skip."""
    idx  = _LETTERS[:Nf]
    kept = idx[f_skip]
    parts, ops = [idx], [L]
    for ff in range(Nf):
        if ff != f_skip:
            parts.append(idx[ff])
            ops.append(xq[ff])
    return np.einsum(','.join(parts) + '->' + kept, *ops)


def _marginalise_A(A, xq):
    """Contract A (No x Ns0 x ... x NsNf-1) against ALL state marginals."""
    result = np.array(A, dtype=float)
    for xf in xq:
        result = np.tensordot(result, np.asarray(xf, dtype=float),
                              axes=[[1], [0]])
    return result.ravel()


def _marginalise_H(H_A, xq, Nf):
    """Contract H_A (Ns0x...xNsNf-1) against all state marginals -> scalar."""
    idx   = _LETTERS[:Nf]
    parts = [idx] + [idx[ff] for ff in range(Nf)]
    ops   = [H_A] + list(xq)
    return float(np.einsum(','.join(parts) + '->', *ops))


# ---------------------------------------------------------------------------
# Phase 1 — Variational free energy minimisation over hidden states
# ---------------------------------------------------------------------------

def minimise_F(x, L, D, sB, rB, V, t, S, Nf, Ns, Np, tau, ni, policies=None):
    """
    Minimise variational free energy over posterior beliefs.

    Implements Eqs 24-26 of the paper:
        epsilon_{pi,tau} = 1/2(ln B s_{tau-1}) + 1/2(ln B^T s_{tau+1})
                         + ln A^T o_tau  -  ln s_{pi,tau}
        v_{pi,tau}  <- v_{pi,tau} + epsilon_{pi,tau}
        s_{pi,tau}  <- sigma(v_{pi,tau})

    The update is a damped fixed-point iteration in log-space.
    Convergence is guaranteed because the Jacobian eigenvalues are
    (1 - 1/tau) < 1 for tau > 0.5, making the fixed point attracting.
    The fixed point equals the minimum of F (strictly convex KL divergence).
    """
    if policies is None:
        policies = list(range(Np))

    F = np.zeros(Np)
    G = np.zeros(Np)
    xn = [np.zeros((ni, Ns[f], S, Np)) for f in range(Nf)]
    vn = [np.zeros((ni, Ns[f], S, Np)) for f in range(Nf)]

    for k in policies:
        dF = 1.0
        for i in range(ni):
            F[k] = 0.0
            for j in range(S):
                for f in range(Nf):
                    sx = x[f][:, j, k].copy()
                    v  = np.zeros(Ns[f])
                    qL = np.zeros(Ns[f])

                    if dF > np.exp(-8) or i > 4:

                        # ── ln A^T o_tau: likelihood message ──────────────
                        if j <= t and L[j] is not None:
                            xq  = [x[ff][:, j, k] for ff in range(Nf)]
                            res = _marginalise_L_except_f(L[j], xq, f, Nf)
                            qL  = spm_log(np.maximum(res.ravel(), 1e-16))

                        # current log-belief: ln s_{pi,tau}
                        qx = spm_log(sx)

                        # ── forward message: ln B s_{tau-1} ───────────────
                        if j == 0:
                            # at t=0 prior D replaces forward transition
                            v = v + spm_log(D[f])
                        else:
                            u_idx = int(V[j - 1, k, f]) - 1
                            px = spm_log(sB[f][:, :, u_idx] @ x[f][:, j - 1, k])
                            v  = v + px

                        # ── backward message: ln B^T s_{tau+1} ────────────
                        if j < S - 1:
                            u_idx = int(V[j, k, f]) - 1
                            px = spm_log(rB[f][:, :, u_idx] @ x[f][:, j + 1, k])
                            v  = v + px

                        # ── likelihood + normalisation — added exactly once ──
                        # Eq 24: ε = ½ ln B s_{τ−1} + ½ ln B^T s_{τ+1}
                        #            + ln A^T o_τ  −  ln s_{π,τ}
                        # qL is the zero vector for future timesteps (j > t),
                        # so this is safe to apply unconditionally.
                        v = v + qL - qx

                        # ── VFE contribution (uniform ½ factor) ───────────
                        # −VFE ≈ sx·v; ½ normalises transition terms that are
                        # counted from both sides.  No special case needed now
                        # that qL−qx is no longer double-counted.
                        F[k] += float(sx @ (0.5 * v))

                        # ── damped fixed-point update (Eq 26) ─────────────
                        # ln s <- ln s + v/tau  then project onto simplex
                        v  = v - v.mean()
                        sx = spm_softmax(qx + v / tau)

                    else:
                        F[k] = G[k]   # convergence shortcut

                    x[f][:, j, k]     = sx
                    xn[f][i, :, j, k] = sx    # neural state encoding
                    vn[f][i, :, j, k] = v     # prediction error trace (ERP)

            if i > 0:
                dF = float(F[k] - G[k])
            G = F.copy()

    return x, F, xn, vn


# ---------------------------------------------------------------------------
# Phase 2 — Expected free energy
# ---------------------------------------------------------------------------

def compute_expected_G(x, A, a, d, C, t, S, Nf, Ng, Np, policies=None):
    """
    Expected Free Energy per policy — Eqs 27 and 39 of the paper.

    G_pi = sum_tau  zeta_{pi,tau}

    Outcome prediction error (Eq 27):

        zeta_{pi,tau} =   As_{pi,tau} * (ln As_{pi,tau} - ln C_tau)   [risk]
                        - diag(A^T ln A) * s_{pi,tau}                  [ambiguity]
                        - As_{pi,tau} * W_A * s_{pi,tau}               [A-novelty]

    Plus D-novelty term (when learning d, analogous to Eq 38-39):

        - w_D * s_{pi, tau=0}                                          [D-novelty]

    Notes
    -----
    - zeta is computed ONCE per policy per timestep — NOT iterated.
    - G minimisation happens via policy selection in update_policy_posterior.
    - C[g] is already stored as log-preferences (ln C) by solver.py.
    - ambiguity carries MINUS sign  (Eq 27: subtracted).
    - A-novelty carries MINUS sign  (Eq 39: subtracted).
    - D-novelty carries MINUS sign  (analogous).
    - spm_wnorm is already imported at module level — do NOT re-import inside loop.

    Parameters
    ----------
    x  : list of arrays x[f] shape (Ns_f, S, Np) — posterior beliefs
    A  : list of arrays A[g] shape (No, Ns0, ...) — likelihood (normalised)
    a  : list of arrays or None — Dirichlet params for A (for W_A novelty)
    d  : list of arrays or None — Dirichlet params for D (for W_D novelty)
    C  : list of arrays C[g] shape (No, T)        — log-preferences (ln C)
    t  : int — current timestep (sum G from t to S)
    S  : int — total timesteps
    Nf : int — number of hidden state factors
    Ng : int — number of observation modalities
    Np : int — number of policies
    """
    if policies is None:
        policies = list(range(Np))

    G = np.zeros(Np)

    for k in policies:

        # ── D-novelty: −w_D * s_{pi, tau=0} ──────────────────────────────
        # measures uncertainty about initial state prior d
        # uses beliefs at timestep 0 only
        if d is not None:
            for f in range(Nf):
                if d[f] is not None:
                    wD_f = spm_wnorm(d[f].reshape(-1, 1)).ravel()
                    G[k] -= float(np.dot(wD_f, x[f][:, 0, k]))

        for j in range(t, S):

            # beliefs over states at timestep j under policy k
            # xq[f] shape: (Ns_f,)
            xq = [x[f][:, j, k] for f in range(Nf)]

            for g in range(Ng):

                # ── qo = As_{pi,tau} ──────────────────────────────────────
                # predicted observation distribution
                # qo[o] = sum_{s0,...} A[o,s0,...] * q(s0) * ...
                # shape: (No,)
                qo = _marginalise_A(A[g], xq)

                # ── ln C_tau: log-preference at timestep j ────────────────
                # C[g] is already ln C (preprocessed by solver.py)
                # shape: (No,)
                ln_C = C[g][:, j] if C[g].ndim == 2 else C[g]

                # ── Term 1: RISK  As*(ln As - ln C) ──────────────────────
                # = KL[qo || C] (up to normalisation)
                # scalar
                risk = float(qo @ (spm_log(qo) - ln_C))

                # ── Term 2: AMBIGUITY  −diag(A^T ln A) * s ───────────────
                # diag(A^T ln A)[s] = sum_o A[o,s] * ln A[o,s]  per state s
                # shape before contraction: (Ns0, Ns1, ...)
                # MINUS sign: subtracted in Eq 27
                diag_AtlnA = np.sum(A[g] * spm_log(A[g]), axis=0)
                ambiguity  = float(_marginalise_H(diag_AtlnA, xq, Nf))

                # ── Term 3: A-NOVELTY  −As * W_A * s ─────────────────────
                # only present when learning A (Dirichlet params a provided)
                # W_A = spm_wnorm(a) measures parameter uncertainty
                # MINUS sign: subtracted in Eq 39
                novelty = 0.0
                if a is not None and a[g] is not None:
                    W_A     = spm_wnorm(a[g]) * (a[g] > 0)  # mask zero entries
                    Ws      = _marginalise_A(W_A, xq)        # shape: (No,)
                    novelty = float(qo @ Ws)

                # ── zeta_{pi,tau}: outcome prediction error (Eq 27) ───────
                # computed ONCE — not iteratively minimised
                # G_pi = sum_tau zeta_{pi,tau}
                zeta = risk - ambiguity - novelty
                G[k] += zeta

    # G is returned as raw EFE (higher = worse policy)
    # policy selection in update_policy_posterior uses: q(pi) = sigma(-w*G + ...)
    return G


# ---------------------------------------------------------------------------
# Phase 3 — Policy posterior + precision
# ---------------------------------------------------------------------------

def update_policy_posterior(Q, F, E, beta_init, Ni_iters, fix_precision=False):
    """
    Posterior over policies and precision.

    q(pi) = sigma( ln E  -  w*G  +  F )

    where:
        Q   = G = Expected Free Energy per policy (from compute_expected_G)
                  higher Q = worse policy — enters with MINUS sign
        F   = VFE per policy from minimise_F — policies explaining current
              observations better score higher F
        E   = log prior over policies (habits)
        w   = 1/beta = precision

    Precision beta is inferred by gradient descent on its own free energy:

        dF/d_beta = beta - beta_0  +  E_q[(qu - pu) * G]

    where:
        qu = sigma(ln E - w*G + F)   posterior (includes current VFE F)
        pu = sigma(ln E - w*G)       prior     (excludes F)

    Parameters
    ----------
    Q          : array (Np,) — EFE per policy (from compute_expected_G)
    F          : array (Np,) — VFE per policy (from minimise_F)
    E          : array (Np,) — log prior over policies (habits)
    beta_init  : float       — prior on beta (inverse precision)
    Ni_iters   : int         — gradient descent steps on beta
    fix_precision : bool     — if True keep w = 1/beta_init fixed
    """
    qb = beta_init
    w  = 1.0 / qb

    for _ in range(Ni_iters):

        # posterior: G enters with MINUS sign (minimised)
        qu = spm_softmax(E - w * Q + F)

        # prior: same but without current VFE F
        pu = spm_softmax(E - w * Q)

        if not fix_precision:

            # expected G under posterior vs prior
            # how much does F shift policy weights?
            eg = float((qu - pu) @ Q)

            # gradient of free energy w.r.t. beta
            dFdg = qb - beta_init + eg

            # gradient descent step
            qb = qb - dFdg / 2.0

            # beta must stay positive
            qb = max(qb, 1e-6)

            # recover precision
            w = 1.0 / qb

    # final posterior with converged precision
    qu = spm_softmax(E - w * Q + F)

    return qu, w, qb


def _marginalise_scalar(H, xq):
    result = np.array(H, dtype=float)
    for xf in xq:
        result = np.tensordot(result, np.asarray(xf, dtype=float),
                              axes=[[0], [0]])
    return float(np.asarray(result).ravel().sum())
