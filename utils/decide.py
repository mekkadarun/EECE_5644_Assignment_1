# utils/decide.py
import numpy as np
from utils.gaussian import log_gaussian

def class_likelihoods(X, means, covs):
    """p(x|L=j) for the 3 Q2 classes, class 3 is an equal mix of components 3 and 4.

    Returns an array of shape (3, N): one row per class, one column per sample.
    """

    #  Single gaussians
    p1 = np.exp(log_gaussian(X, means[0], covs[0]))
    p2 = np.exp(log_gaussian(X, means[1], covs[1]))
    #  Class 3 - 50/50 mixture of two gaussians
    p3a = np.exp(log_gaussian(X, means[2], covs[2]))
    p3b = np.exp(log_gaussian(X, means[3], covs[3]))
    p3 = 0.5 * p3a + 0.5 * p3b

    return np.vstack([p1, p2, p3])

def posteriors(lik, priors):
    """Bayes' rule: turn a (C, N) likelihood table into posteriors P(L=j|x)."""

    joint = lik * np.array(priors)[:, None]
    return joint / joint.sum(axis=0)

def erm_decide(post, loss):
    """Minimum expected risk decision. loss[i, j] = cost of deciding i when the truth is j."""

    risk = np.array(loss) @ post
    return np.argmin(risk, axis=0)

def confusion_matrix(D, L, n_classes):
    """Count (decision i, true label j) pairs. Returns the counts and P(D=i|L=j)."""

    #  Start with a table of zeros
    counts = np.zeros((n_classes, n_classes))
    # Loop through every decision-label pair
    for i in range(n_classes):
        for j in range(n_classes):
            counts[i, j] = np.sum((D == i) & (L == j))
    return counts, counts / counts.sum(axis=0)

def map_decide_log(X, priors, means, covs):
    """MAP decision in log space: argmax over c of ln g(x|mean_c, cov_c) + ln P(c).
    """
    # Compute log-likelihood + log-prior for each sample
    scores = []
    for c in range(len(priors)):
        scores.append(log_gaussian(X, means[c], covs[c]) + np.log(priors[c]))
    # Turn the list into a table with one row per class
    scores = np.vstack(scores)
    return np.argmax(scores, axis=0)
