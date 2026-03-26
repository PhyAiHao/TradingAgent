"""
mdp_model.py
------------
MDPModel dataclass — holds all matrices and parameters for one level
of the hierarchical active inference model.

Capital letters  (A, B, D)  = generative process  (the world as it is)
Lowercase letters (a, b, d)  = generative model    (agent's learnable beliefs)

Attribute naming convention (mirrors MATLAB SPM):
  A, B, D     — generative process matrices (fixed, not learned)
  a, b, d, e  — concentration parameters (learned, lowercase)
  a_0, b_0, d_0, e_0  — frozen baselines for the forgetting rule
  C           — fixed log-preferences (NOT learned in this model)
                There is intentionally no C_0 — C is never updated.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import copy


@dataclass
class MDPModel:
    # ------------------------------------------------------------------
    # Generative process  (uppercase — fixed, not learned)
    # ------------------------------------------------------------------
    A: List[np.ndarray]          # A[g]: likelihood tensor (No × Ns0 × ... × NsNf-1)
    B: List[np.ndarray]          # B[f]: transition tensor (Ns × Ns × Nu)
    D: List[np.ndarray]          # D[f]: prior over initial states (Ns,)
    T: int = 6                   # number of timesteps

    # ------------------------------------------------------------------
    # Generative model  (lowercase — learned via Dirichlet updates)
    # ------------------------------------------------------------------
    a: Optional[List[np.ndarray]] = None   # concentration params for A
    b: Optional[List[np.ndarray]] = None   # concentration params for B
    d: Optional[List[np.ndarray]] = None   # concentration params for D
    e: Optional[np.ndarray]       = None   # concentration params for policies

    # Frozen baselines used in the forgetting rule:
    #   θ ← (θ − θ_0)(1 − ω) + θ_0 + η·δθ
    a_0: Optional[List[np.ndarray]] = None
    b_0: Optional[List[np.ndarray]] = None
    d_0: Optional[List[np.ndarray]] = None
    e_0: Optional[np.ndarray]       = None

    # ------------------------------------------------------------------
    # Fixed preferences  (uppercase C — NOT learned, no C_0 baseline)
    # ------------------------------------------------------------------
    C: Optional[List[np.ndarray]] = None   # C[g]: log-preferences (No[g] × T)

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------
    V: Optional[np.ndarray] = None   # shape (T-1, n_policies, n_factors)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    alpha: float = 512.0   # action precision
    beta:  float = 1.0     # prior precision over policies
    eta:   float = 1.0     # learning rate
    omega: float = 1.0     # forgetting rate  (1 = no forgetting)
    tau:   float = 4.0     # gradient-descent time constant
    erp:   float = 1.0     # belief-reset parameter (ERP simulation)
    zeta:  float = 3.0     # Occam window for pruning unlikely policies
    chi:   float = 1/64    # Occam window for hierarchical updates
    ni:    int   = 32      # number of VB gradient-descent iterations

    # ------------------------------------------------------------------
    # Hierarchical link to subordinate level
    # ------------------------------------------------------------------
    MDP:  Optional['MDPModel'] = None   # subordinate MDP (Level 1)
    link: Optional[np.ndarray] = None   # link matrix (n_sub_factors × n_L2_modalities)

    # ------------------------------------------------------------------
    # True states / outcomes  (set by simulator at run time)
    # ------------------------------------------------------------------
    s: Optional[np.ndarray] = None   # (Nf, T)   true hidden states
    o: Optional[np.ndarray] = None   # (Ng, T)   observed outcomes
    u: Optional[np.ndarray] = None   # (Nf, T-1) actions taken

    # ------------------------------------------------------------------
    # Outputs written back after solve
    # ------------------------------------------------------------------
    X:     Optional[List[np.ndarray]] = None   # Bayesian model-average beliefs
    Q:     Optional[List[np.ndarray]] = None   # per-policy beliefs
    R:     Optional[np.ndarray]       = None   # policy posteriors  (Np, T)
    F:     Optional[np.ndarray]       = None   # variational free energy  (Np, T)
    G:     Optional[np.ndarray]       = None   # expected free energy     (Np, T)
    H:     Optional[np.ndarray]       = None   # total free energy        (T,)
    w:     Optional[np.ndarray]       = None   # precision trace          (T,)
    vn:    Optional[list]             = None   # neuronal prediction errors (ERP)
    xn:    Optional[list]             = None   # neuronal state encoding
    Fa:    Optional[dict]             = None   # free energy of a params
    Fb:    Optional[dict]             = None   # free energy of b params
    Fd:    Optional[dict]             = None   # free energy of d params
    mdp_t: Optional[list]             = None   # per-timestep subordinate MDP results

    def __post_init__(self):
        """Freeze baseline copies of learnable concentration parameters."""
        if self.a is not None and self.a_0 is None:
            self.a_0 = [x.copy() for x in self.a]
        if self.b is not None and self.b_0 is None:
            self.b_0 = [x.copy() for x in self.b]
        if self.d is not None and self.d_0 is None:
            self.d_0 = [x.copy() for x in self.d]
        if self.e is not None and self.e_0 is None:
            self.e_0 = self.e.copy()
        # NOTE: C has no baseline copy (C_0) because preferences are
        # fixed, not learned. Do not add C_0 here.

    def copy(self):
        return copy.deepcopy(self)
