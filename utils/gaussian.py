# utils/gaussian.py

from scipy.stats import multivariate_normal
import numpy as np

def log_gaussian(X, m, C):
    return multivariate_normal.logpdf(X, mean=m, cov=C)

def fit_gaussian_classes(X, y, alpha, shared=False):
    """Estimate prior, mean and regularized covariance for each class.

    y holds class indices 0..C-1. Regularization: C_reg = C + lam*I,
    with lam = alpha * trace(C) / rank(C)
    shared=True uses the average lambda for all classes.
    Returns lists: priors, means, regularized covariances, lambdas.
    """

    # Largest index + 1 - number of classes, d - number of features
    n_classes = y.max() + 1
    d = X.shape[1]
    priors, means, covs, lambdas = [], [], [], []
    # Loop over the classes
    for c in range(n_classes):
        Xc = X[y == c]
        # Priors - class's share of the samples and mean - average of each feature
        priors.append(len(Xc) / len(X))
        means.append(Xc.mean(axis=0))
        #  Sample covariance
        C = np.cov(Xc, rowvar=False)
        #  Formula in the hint
        lam = alpha * np.trace(C) / np.linalg.matrix_rank(C)
        # Store the raw covariance for now; lambda is added after the loop
        covs.append(C)
        lambdas.append(lam)

    # Replace every class's lambda with their average
    if shared:
        lambdas = [np.mean(lambdas)] * n_classes

    # Add each class's lambda to every diagonal entry of its covariance
    for c in range(n_classes):
        covs[c] = covs[c] + lambdas[c] * np.eye(d)

    return priors, means, covs, lambdas
