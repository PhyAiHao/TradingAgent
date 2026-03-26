"""
solver.py  —  General-purpose active inference solver
=====================================================
Implements the variational message passing loop from
spm_MDP_VB_X_tutorial.m (Smith, Friston, Whyte).

This file is model-agnostic. All hyperparameters are read from the
MDPModel instance passed in — nothing is imported from config.py.

Parameter treatment (matches spm_MDP_VB_X_tutorial.m exactly):
─────────────────────────────────────────────────────────────────
All model matrices use spm_norm (E[θ] = a/Σa), NOT digamma.
This is confirmed by reading the tutorial MATLAB source directly:

  Line 278:  A{m,g}  = spm_norm(MDP(m).a{g})   ← E[A]
  Line 304:  sB{m,f} = spm_norm(MDP(m).b{f})   ← E[B]
  Line 325:  D{m,f}  = spm_norm(MDP(m).d{f})   ← E[D]
  Line 344:  E{m}    = spm_norm(MDP(m).e)       ← E[E]
  Line 350:  qE{m}   = spm_log(E{m})            ← ln(E[E])

spm_KL_dir is used ONLY for the output complexity costs (Fa, Fb, Fd, Fe)
at lines 1179-1200 — AFTER learning, as a diagnostic output — NOT during
the inference loop. This is already correctly implemented in learning.py.

Hierarchical link notes:
  FIX 1 — Sub-MDP D prior uses beliefs BEFORE current timestep's VB.
    MATLAB line 587: O = spm_dot(A, xq) where xq is the belief vector
    from the PREVIOUS outer timestep's VB iterations.
  FIX 2 — L2 receives SOFT outcome from sub-MDP, not a one-hot sample.
    MATLAB line 653: O{g,t} = link' * mdp(t).X{f}(:,1)
"""
import numpy as np
import copy

from .utils     import spm_log, spm_norm, spm_softmax, spm_wnorm, sample_categorical
from .inference import minimise_F, compute_expected_G, update_policy_posterior, _marginalise_A
from .learning  import update_parameters


def spm_MDP_VB_X(mdp):
    """
    Solve one trial of an active inference MDP.

    All hyperparameters are read from mdp:
      mdp.ni    — VB iterations (default 32)
      mdp.tau   — gradient-descent time constant
      mdp.alpha — action precision
      mdp.beta  — prior policy precision
      mdp.erp   — belief-reset parameter
    """
    Ni = mdp.ni

    Ng = len(mdp.A)
    Nf = len(mdp.B)
    T  = mdp.T
    V  = mdp.V
    Np = V.shape[1] if V is not None else 1

    for f in range(Nf):
        if mdp.B[f].ndim == 2:
            mdp.B[f] = mdp.B[f][:, :, np.newaxis]

    Ns = [mdp.B[f].shape[0] for f in range(Nf)]
    Nu = [mdp.B[f].shape[2] for f in range(Nf)]
    No = [mdp.A[g].shape[0] for g in range(Ng)]
    S  = T

    # ------------------------------------------------------------------
    # sB / rB — transition matrices for message passing.
    # MATLAB line 304: sB = spm_norm(b)  when b exists, else spm_norm(B).
    # spm_norm gives E[B] = b/Σb. sB is multiplied by beliefs directly
    # inside minimise_F — no log of sB itself is taken.
    # ------------------------------------------------------------------
    sB, rB = [], []
    for f in range(Nf):
        B_src = mdp.b[f] if mdp.b is not None else mdp.B[f]
        sB.append(np.stack(
            [spm_norm(B_src[:, :, j])   for j in range(Nu[f])], axis=2))
        rB.append(np.stack(
            [spm_norm(B_src[:, :, j].T) for j in range(Nu[f])], axis=2))

    # ------------------------------------------------------------------
    # A_model — likelihood matrix for generative MODEL.
    # MATLAB line 278: A{m,g} = spm_norm(MDP(m).a{g})  i.e. E[A] = a/Σa
    # MATLAB line 280: A{m,g} = spm_norm(MDP(m).A{g})  when no a.
    # ------------------------------------------------------------------
    if mdp.a is not None:
        A_model = [spm_norm(mdp.a[g]) for g in range(Ng)]
    else:
        A_model = [spm_norm(mdp.A[g]) for g in range(Ng)]

    # ------------------------------------------------------------------
    # D_model — initial state prior for generative MODEL.
    # MATLAB line 325: D{m,f} = spm_norm(MDP(m).d{f})  i.e. E[D] = d/Σd
    # MATLAB line 327: D{m,f} = spm_norm(MDP(m).D{f})  when no d.
    # minimise_F calls spm_log(D[f]) at line 92 — so D_model is a
    # probability vector and spm_log gives ln(E[D]).
    # ------------------------------------------------------------------
    D_model = [spm_norm(mdp.d[f]) if mdp.d is not None
               else spm_norm(mdp.D[f]) for f in range(Nf)]

    # ------------------------------------------------------------------
    # E — log prior over policies (habits).
    # MATLAB line 344: E{m}  = spm_norm(MDP(m).e)   i.e. E[E] = e/Σe
    # MATLAB line 350: qE{m} = spm_log(E{m})         i.e. ln(E[E])
    # update_policy_posterior expects E in log-space.
    # ------------------------------------------------------------------
    if mdp.e is not None:
        E = spm_log(spm_norm(mdp.e.ravel()))
    else:
        E = spm_log(np.ones(Np) / Np)

    # ------------------------------------------------------------------
    # Parameter uncertainty matrices for EFE novelty terms (Eq 40).
    # MATLAB line 287: wA = spm_wnorm(a).*(a > 0)
    # MATLAB line 337: wD = spm_wnorm(d)
    # wA and wD are used in compute_expected_G for the novelty term.
    # These are NOT used in the inference loop directly.
    # ------------------------------------------------------------------
    wA = [spm_wnorm(mdp.a[g]) * (mdp.a[g] > 0)
          if mdp.a is not None else None for g in range(Ng)]
    wD = [spm_wnorm(mdp.d[f].reshape(-1, 1)).ravel()
          if mdp.d is not None else None for f in range(Nf)]

    # ------------------------------------------------------------------
    # Log-preferences C.
    # MATLAB line 362: C = spm_psi(c + 1/32)  when learnable c exists.
    # MATLAB line 365: C = MDP.C{g}           when fixed C given.
    # MATLAB line 379: C = spm_log(spm_softmax(C))  applied to both.
    # Our model uses fixed uppercase C (raw scores), no learnable c.
    # We apply: ln C = spm_log(spm_norm(exp(C_raw)))
    # ------------------------------------------------------------------
    C = []
    for g in range(Ng):
        if mdp.C is not None and len(mdp.C) > g and mdp.C[g] is not None:
            Cg = np.array(mdp.C[g], dtype=float)
            if Cg.ndim == 1:
                Cg = np.tile(Cg[:, None], (1, T))
            if Cg.shape[1] < T:
                Cg = np.concatenate(
                    [Cg, np.tile(Cg[:, -1:], (1, T - Cg.shape[1]))], axis=1)
            C.append(spm_log(spm_norm(np.exp(Cg))))
        else:
            C.append(np.zeros((No[g], T)))

    # ------------------------------------------------------------------
    # Initialise beliefs x[f] shape (Ns_f, T, Np).
    # t=0 column seeded with D_model (prior over initial states).
    # ------------------------------------------------------------------
    x = []
    for f in range(Nf):
        xf = np.ones((Ns[f], T, Np)) / Ns[f]
        for k in range(Np):
            xf[:, 0, k] = D_model[f]
        x.append(xf)
    X = [np.ones((Ns[f], T)) / Ns[f] for f in range(Nf)]

    s         = np.zeros((Nf, T), dtype=int)
    o         = np.zeros((Ng, T), dtype=int)
    u_actions = np.zeros((Nf, T - 1), dtype=int)

    # Pre-populate true states (1-based → 0-based)
    if mdp.s is not None:
        pre_s = np.asarray(mdp.s).ravel()
        for f in range(min(len(pre_s), Nf)):
            if int(pre_s[f]) != 0:
                s[f, 0] = int(pre_s[f]) - 1

    # Pre-populate observations (1-based → 0-based)
    if mdp.o is not None:
        pre_o = np.asarray(mdp.o)
        for g in range(min(pre_o.shape[0], Ng)):
            for t_pre in range(min(pre_o.shape[1], T)):
                if int(pre_o[g, t_pre]) != 0:
                    o[g, t_pre] = int(pre_o[g, t_pre]) - 1

    O             = [[None] * T for _ in range(Ng)]
    L             = [None] * T
    qb            = float(mdp.beta)
    w_trace       = np.zeros(T)
    u_post        = np.ones((Np, T)) / Np
    xn_all        = [np.zeros((Ni, Ns[f], S, T, Np)) for f in range(Nf)]
    vn_all        = [np.zeros((Ni, Ns[f], S, T, Np)) for f in range(Nf)]
    F_all         = np.zeros((Np, T))
    G_all         = np.zeros((Np, T))
    H_all         = np.zeros(T)
    mdp_t_results = [None] * T

    for t in range(T):

        # ------------------------------------------------------------------
        # Sample true hidden states — generative PROCESS (uppercase A, B, D)
        # ------------------------------------------------------------------
        for f in range(Nf):
            if s[f, t] == 0:
                if t == 0:
                    ps = spm_norm(mdp.D[f].ravel())
                else:
                    act = int(u_actions[f, t - 1])
                    ps  = spm_norm(mdp.B[f][:, s[f, t - 1], act].ravel())
                s[f, t] = sample_categorical(ps)

        # ------------------------------------------------------------------
        # Sample observations — generative PROCESS (uppercase A)
        # ------------------------------------------------------------------
        for g in range(Ng):
            if o[g, t] == 0:
                ind     = tuple(s[:, t])
                po      = spm_norm(mdp.A[g][(slice(None),) + ind].ravel())
                o[g, t] = sample_categorical(po)
            oh         = np.zeros(No[g])
            oh[o[g,t]] = 1.0
            O[g][t]    = oh

        # ------------------------------------------------------------------
        # Hierarchical: run subordinate MDP and inject soft L2 outcomes
        # ------------------------------------------------------------------
        if mdp.link is not None and mdp.MDP is not None:
            sub_mdp       = copy.deepcopy(mdp.MDP)
            link          = np.atleast_2d(mdp.link)
            n_sub_factors = link.shape[0]
            n_L2_mods     = link.shape[1]

            # FIX 1: use pre-VB beliefs (X not yet updated this timestep)
            for f_sub in range(n_sub_factors):
                for g_L2 in range(n_L2_mods):
                    if link[f_sub, g_L2] == 1:
                        O_pred = np.asarray(
                            _marginalise_A(A_model[g_L2],
                                           [X[ff][:, t] for ff in range(Nf)]),
                            dtype=float).ravel()
                        sub_mdp.D[f_sub] = spm_norm(O_pred)

            sub_mdp.s        = np.array([o[0, t] + 1])
            sub_out          = spm_MDP_VB_X(sub_mdp)
            mdp_t_results[t] = sub_out

            # FIX 2: soft posterior — no one-hot sampling
            for f_sub in range(n_sub_factors):
                for g_L2 in range(n_L2_mods):
                    if link[f_sub, g_L2] == 1:
                        sub_X  = sub_out.X[f_sub][:, 0]
                        n_out  = No[g_L2]
                        mapped = np.zeros(n_out)
                        for oi in range(min(n_out, len(sub_X))):
                            mapped[oi] = sub_X[oi]
                        O[g_L2][t] = spm_norm(mapped)

        # ------------------------------------------------------------------
        # Joint likelihood L[t] — uses A_model (generative MODEL).
        # A_model = spm_norm(a) = E[A].  spm_log(A_model) inside
        # minimise_F gives ln(E[A]) — matching MATLAB exactly.
        # ------------------------------------------------------------------
        Lt = np.ones([Ns[f] for f in range(Nf)])
        for g in range(Ng):
            Lt = Lt * np.tensordot(A_model[g], O[g][t], axes=[[0], [0]])
        L[t] = Lt

        # ------------------------------------------------------------------
        # ERP belief reset (MATLAB line 787)
        # ------------------------------------------------------------------
        for f in range(Nf):
            log_x = spm_log(x[f])
            x[f]  = np.exp(log_x / mdp.erp)
            s_x   = x[f].sum(axis=0, keepdims=True)
            s_x[s_x == 0] = 1.0
            x[f] /= s_x

        # ------------------------------------------------------------------
        # Phase 1: VB belief update — minimise variational free energy.
        # ------------------------------------------------------------------
        x, F, xn, vn = minimise_F(
            x, L, D_model, sB, rB, V, t, S, Nf, Ns, Np, mdp.tau, Ni)

        # ------------------------------------------------------------------
        # Phase 2: Expected free energy — Eq 27/39.
        # Passes mdp.a and mdp.d so compute_expected_G computes
        # W = spm_wnorm(a/d) internally for the novelty terms.
        # ------------------------------------------------------------------
        Q = compute_expected_G(
            x, A_model, mdp.a, mdp.d, C, t, S, Nf, Ng, Np)

        # ------------------------------------------------------------------
        # Phase 3: Policy posterior + precision.
        # E = spm_log(spm_norm(e)) = ln(E[E])  matches MATLAB qE = spm_log(E).
        # q(π) = σ(E − w·G + F)  — G enters with MINUS sign.
        # ------------------------------------------------------------------
        qu, w_t, qb = update_policy_posterior(Q, F, E, qb, Ni)

        w_trace[t]   = w_t
        u_post[:, t] = qu
        F_all[:, t]  = F
        G_all[:, t]  = Q

        # Bayesian model average: X[f][:,j] = Σ_k qu[k] · x[f][:,j,k]
        for f in range(Nf):
            for j in range(S):
                X[f][:, j] = x[f][:, j, :] @ qu

        for f in range(Nf):
            xn_all[f][:, :, :, t, :] = xn[f]
            vn_all[f][:, :, :, t, :] = vn[f]

        # ------------------------------------------------------------------
        # Action selection
        # ------------------------------------------------------------------
        if t < T - 1:
            Pu = np.zeros(tuple(Nu))
            for k in range(Np):
                v_idx      = tuple(int(V[t, k, f]) - 1 for f in range(Nf))
                Pu[v_idx] += qu[k]
            Pu_flat = spm_softmax(mdp.alpha * spm_log(Pu.ravel() + 1e-16))
            ind     = np.unravel_index(sample_categorical(Pu_flat), tuple(Nu))
            for f in range(Nf):
                u_actions[f, t] = int(ind[f])

    # ------------------------------------------------------------------
    # Learning — update Dirichlet concentration parameters a, b, d, e.
    # spm_KL_dir is used ONLY here (in update_parameters) for the output
    # complexity costs Fa, Fb, Fd — NOT during the inference loop above.
    # ------------------------------------------------------------------
    mdp, Fa, Fb, Fd = update_parameters(mdp, O, X, u_post, V, T, Nf, Ng, Np)

    # ------------------------------------------------------------------
    # Write outputs back into mdp
    # ------------------------------------------------------------------
    mdp.s     = s + 1
    mdp.o     = o + 1
    mdp.u     = u_actions + 1
    mdp.X     = X
    mdp.Q     = x
    mdp.R     = u_post
    mdp.F     = F_all
    mdp.G     = G_all
    mdp.H     = H_all
    mdp.w     = w_trace
    mdp.vn    = vn_all
    mdp.xn    = xn_all
    mdp.Fa    = Fa
    mdp.Fb    = Fb
    mdp.Fd    = Fd
    mdp.mdp_t = mdp_t_results
    return mdp
