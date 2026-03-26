"""
utils.py  —  SPM helper functions (fixed)
"""
import numpy as np
from scipy.special import digamma, gammaln


def spm_log(x):
    """Safe log — floors at log(1e-16) to avoid -inf."""
    return np.log(np.maximum(np.asarray(x, dtype=float), 1e-16))


def spm_softmax(x, k=1.0):
    """
    Exact translation of MATLAB spm_softmax.
    Normalises over ALL elements globally (not per-axis).
    Optional inverse temperature k (default 1).

    MATLAB code:
        x = k*x;
        x = x - max(x(:));
        x = exp(x);
        x = x / sum(x(:));
    """
    x = np.asarray(x, dtype=float) * k
    x = x - x.max()          # subtract global max for numerical stability
    x = np.exp(x)
    return x / x.sum()        # divide by sum of ALL elements


def spm_norm(x):
    """
    Column-wise normalisation so each column sums to 1.
    This is DIFFERENT from spm_softmax — do not confuse the two.
    """
    x = np.asarray(x, dtype=float)
    s = x.sum(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return x / s


def spm_dot(X, x_list, skip_f=None):
    """
    Multidimensional dot product.
    Contracts likelihood tensor X (outcomes × s0 × … × sF-1)
    against state marginals x_list = [x0, …, xF-1].
    skip_f : leave factor skip_f un-contracted (returns Ns[f]-vector).
    """
    result = np.array(X, dtype=float)
    skipped = 0
    for f, xf in enumerate(x_list):
        if skip_f is not None and f == skip_f:
            skipped += 1
            continue
        result = np.tensordot(result, np.asarray(xf, dtype=float),
                              axes=[[1], [0]])
    return result


def spm_cross(a, b):
    """Outer product of two vectors — used in Dirichlet updates."""
    return np.outer(np.asarray(a).ravel(), np.asarray(b).ravel())


def spm_psi(x):
    """Digamma function."""
    return digamma(np.asarray(x, dtype=float))


def spm_wnorm(A):
    """
    Precision-weighted normalisation for epistemic value of parameters.
    Matches MATLAB spm_wnorm: (1/sum - 1/A) / 2

    Equation 40 in the paper writes W = ½(a^⊙(-1) - a_sums^⊙(-1)),
    i.e. ½(1/a - 1/sum). MATLAB uses the negative convention
    ½(1/sum - 1/a), and the calling code subtracts it, giving the
    same net result. The critical factor is the ½ — previously missing.
    """
    A = np.asarray(A, dtype=float)
    s = A.sum(axis=0, keepdims=True)
    return (1.0 / np.maximum(s, 1e-16) - 1.0 / np.maximum(A, 1e-16)) / 2.0


def spm_KL_dir(q, p):
    """
    KL divergence KL[Dir(q) || Dir(p)].
    Zero-concentration entries are excluded: digamma(0) = -inf and
    gammaln(0) = inf, so 0 * digamma(0) = nan without masking.
    """
    q = np.asarray(q, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    # Only include entries where both q and p are positive
    mask = (q > 0) & (p > 0)
    q_m  = q[mask]
    p_m  = p[mask]
    if len(q_m) == 0:
        return 0.0
    kl = (gammaln(p_m.sum()) - gammaln(q_m.sum())
          + gammaln(q_m).sum() - gammaln(p_m).sum()
          + ((q_m - p_m) * digamma(q_m)).sum())
    return float(kl)


def sample_categorical(p):
    """Draw one sample from categorical distribution p (0-based index)."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    return int(np.searchsorted(np.cumsum(p), np.random.rand()))
