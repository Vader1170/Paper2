import numpy as np

def uniform_mesh(n, a=0, b=1):
    """Equally spaced nodes."""
    return np.linspace(a, b, n+1)

def perturbed_mesh(n, a=0, b=1, eps=0.05):
    """Uniform nodes with small random perturbation."""
    x = np.linspace(a, b, n+1)
    dx = eps * (b-a) / n
    x[1:-1] += np.random.uniform(-dx, dx, n-1)
    x[0] = a
    x[-1] = b
    return np.sort(x)

def clustered_mesh(n, a=0, b=1, cluster_frac=0.3, cluster_width=0.1):
    """Nodes clustered around the centre of the interval."""
    n_clust = int(n * cluster_frac)
    n_rest = n - n_clust
    x_clust = np.random.uniform(0.5 - cluster_width/2, 0.5 + cluster_width/2, n_clust)
    x_left = np.random.uniform(a, 0.5 - cluster_width/2, n_rest // 2)
    x_right = np.random.uniform(0.5 + cluster_width/2, b, n_rest - n_rest//2)
    x = np.concatenate([x_left, x_clust, x_right])
    return np.sort(x)

def geometric_mesh(n, a=0, b=1, R=1.2):
    """Gaps grow geometrically: delta_i = c * R^i."""
    gaps = np.array([R**i for i in range(n)])
    gaps = gaps / np.sum(gaps) * (b-a)
    x = np.zeros(n+1)
    x[0] = a
    for i in range(n):
        x[i+1] = x[i] + gaps[i]
    return x

def balanced_geometric_mesh(n, a=0, b=1, R=1.2):
    """
    Geometric mesh that satisfies the balanced gap condition.
    For R fixed, sum(delta_i^3) = O(1/n^2).
    """
    return geometric_mesh(n, a, b, R)

def exponential_counterexample(n, a=0, b=1, R=1.5):
    """
    Extreme exponential mesh with R chosen so that sum(delta_i^3) does not decay.
    This deliberately violates the balanced gap condition.
    """
    gaps = np.array([R**i for i in range(n)])
    gaps = gaps / np.sum(gaps) * (b-a)
    x = np.zeros(n+1)
    x[0] = a
    for i in range(n):
        x[i+1] = x[i] + gaps[i]
    return x

def large_gap_mesh(n, a=0, b=1, gap_size=0.4):
    """Single large gap followed by fine uniform mesh."""
    gap = gap_size
    rest = (b-a) - gap
    n_fine = n-1
    if n_fine <= 0:
        return np.array([a, b])
    fine_gaps = rest / n_fine
    x = np.zeros(n+1)
    x[0] = a
    x[1] = a + gap
    for i in range(1, n):
        x[i+1] = x[i] + fine_gaps
    return x

def random_mesh(n, a=0, b=1):
    """Fully random nodes drawn from uniform distribution."""
    x = np.random.uniform(a, b, n-1)
    x = np.sort(x)
    return np.concatenate([[a], x, [b]])
