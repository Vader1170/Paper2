import numpy as np
from scipy.special import eval_legendre
from numpy.linalg import cond

def build_gram_matrix(x, weights, m=5):
    """
    Build the (m+1)x(m+1) Gram matrix G_{jk} = sum_i w_i psi_j(x_i) psi_k(x_i)
    where psi_j are Legendre polynomials orthonormal on [0,1].
    """
    n = len(x) - 1
    # Map [0,1] to [-1,1] for Legendre
    s = 2.0 * x - 1.0
    # Precompute polynomial evaluations at all nodes
    psi = np.zeros((n+1, m+1))
    for j in range(m+1):
        # Normalisation: integral_{-1}^{1} P_j^2 = 2/(2j+1), so on [0,1] norm^2 = 1/(2j+1)
        # We want orthonormal on [0,1], so scale by sqrt(2j+1)
        psi[:, j] = eval_legendre(j, s) * np.sqrt(2*j + 1)
    
    G = np.zeros((m+1, m+1))
    for i in range(n+1):
        wi = weights[i]
        if wi > 0:
            G += wi * np.outer(psi[i, :], psi[i, :])
    return G

def condition_number(x, weights, m=5):
    """Compute the 2-norm condition number of the Gram matrix."""
    G = build_gram_matrix(x, weights, m)
    return cond(G, p=2)

def compute_gap_metrics(x):
    """Return h_max, h_avg, and sum_delta_cubed."""
    delta = np.diff(x)
    h_max = np.max(delta)
    h_avg = (x[-1] - x[0]) / len(delta)
    sum_cubed = np.sum(delta**3)
    return h_max, h_avg, sum_cubed
