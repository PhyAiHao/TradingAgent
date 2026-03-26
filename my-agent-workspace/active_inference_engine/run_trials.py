"""
run_trials.py
-------------
Multi-trial loop — equivalent to the deal() + sequential parameter-update
loop in spm_MDP_VB_X_tutorial.m (lines 164–213).

Passes learned concentration parameters (a, b, d, e) forward from
trial i to trial i+1, implementing trial-by-trial learning.
"""

import copy
from .solver import spm_MDP_VB_X


def run_trials(mdp_template, N):
    """
    Run N sequential trials, carrying learned parameters forward.

    Parameters
    ----------
    mdp_template : MDPModel  — base model specification
    N            : int       — number of trials

    Returns
    -------
    results : list of N MDPModel objects (one per trial, with outputs)
    """
    results = []

    for i in range(N):
        # deep-copy the template for this trial
        mdp = copy.deepcopy(mdp_template)

        # carry forward concentration parameters from the previous trial
        if i > 0:
            prev = results[i - 1]
            mdp  = _update_params_from_previous(mdp, prev)

        out = spm_MDP_VB_X(mdp)
        results.append(out)
        print(f"  Trial {i + 1}/{N} complete")

    return results


def _update_params_from_previous(mdp, prev):
    """
    Copy learned concentration parameters from *prev* trial into *mdp*.
    Equivalent to spm_MDP_update() in the MATLAB code.
    """
    if prev.a is not None and mdp.a is not None:
        mdp.a   = [p.copy() for p in prev.a]
        mdp.a_0 = [p.copy() for p in prev.a_0]   # keep original baselines

    if prev.b is not None and mdp.b is not None:
        mdp.b   = [p.copy() for p in prev.b]
        mdp.b_0 = [p.copy() for p in prev.b_0]

    if prev.d is not None and mdp.d is not None:
        mdp.d   = [p.copy() for p in prev.d]
        mdp.d_0 = [p.copy() for p in prev.d_0]

    if prev.e is not None and mdp.e is not None:
        mdp.e   = prev.e.copy()
        mdp.e_0 = prev.e_0.copy()

    # carry sub-level parameters too (hierarchical learning)
    if (prev.mdp_t is not None and mdp.MDP is not None
            and any(r is not None for r in prev.mdp_t)):
        last_sub = next(
            (r for r in reversed(prev.mdp_t) if r is not None), None
        )
        if last_sub is not None:
            if last_sub.a is not None and mdp.MDP.a is not None:
                mdp.MDP.a   = [p.copy() for p in last_sub.a]
                mdp.MDP.a_0 = [p.copy() for p in last_sub.a_0]
            if last_sub.d is not None and mdp.MDP.d is not None:
                mdp.MDP.d   = [p.copy() for p in last_sub.d]
                mdp.MDP.d_0 = [p.copy() for p in last_sub.d_0]

    return mdp
