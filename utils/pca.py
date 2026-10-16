# utils/pca.py

import numpy as np

def pca(X):
    """Principal component analysis. Returns all projections Z (N, d), sorted by
    variance, and the fraction of total variance each component explains."""
    # Centering each feature's mean
    Xc = X - X.mean(axis=0)
    # Eigenvectors of the covariance - principal directions
    # Eigenvalues are the variances along them
    vals, vecs = np.linalg.eigh(np.cov(Xc, rowvar=False))

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    explained = vals / vals.sum()
    return Xc @ vecs, explained
