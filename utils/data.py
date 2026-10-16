# utils/data.py

import numpy as np
import os
import pandas as pd


def sample_gaussian_mixture(priors, means, covs, n_samples, seed=None):

    # Draw samples from a mixture of Gaussians, one Gaussian per class.

    # Inputs
    #     priors    : list of class probabilities, e.g. [0.35, 0.65]
    #     means     : list of mean vectors, one per class
    #     covs      : list of covariance matrices, one per class
    #     n_samples : how many samples to generate
    #     seed      : fixes the randomness so results are reproducible

    # Outputs
    #     X : array of shape (n_samples, d), the points
    #     L : array of shape (n_samples,), the true label of each point

    rng = np.random.default_rng(seed)
    n_classes = len(priors)
    dim = len(means[0])

    # true labels
    L = rng.choice(n_classes, size=n_samples, p=priors)

    # draw each sample from its own class's Gaussian
    X = np.zeros((n_samples, dim))
    for c in range(n_classes):
        idx = (L == c)
        X[idx] = rng.multivariate_normal(means[c], covs[c], size=idx.sum())

    return X, L


#  Save points and labels together in one .npz file
def save_data(path, X, L):
    np.savez(path, X=X, L=L)

# Load points and labels saved by save_data
def load_data(path):
    d = np.load(path)
    return d["X"], d["L"]

def load_wine(path):
    """Load the white wine CSV. Returns features X (N, 11), quality labels L (N,), and feature names."""

#  Read the file into a pandas DataFrame
    df = pd.read_csv(path, sep=";")
# Drop "quality" columns & keep the 11 features
    X = df.drop(columns="quality").to_numpy()
    L = df["quality"].to_numpy()
#  Used to label plot axes later
    names = list(df.columns[:-1])
    return X, L, names

def load_har(folder):
    """Load the activity dataset: stack train and test into X (10299, 561) and labels L (10299,)."""

#  Read a text file into an array: each row is one sample with 561 numbers
    X_train = np.loadtxt(os.path.join(folder, "train", "X_train.txt"))
    X_test = np.loadtxt(os.path.join(folder, "test", "X_test.txt"))
#  Label files: one number from 1 to 6 per line
    y_train = np.loadtxt(os.path.join(folder, "train", "y_train.txt"), dtype=int)
    y_test = np.loadtxt(os.path.join(folder, "test", "y_test.txt"), dtype=int)
# Stacking two feature tables: 10,299 rows and concatenate the lists end-to-end
    X = np.vstack([X_train, X_test])
    L = np.concatenate([y_train, y_test])
# Store labels under names
    labels = np.loadtxt(os.path.join(folder, "activity_labels.txt"), dtype=str)
    names = list(labels[:, 1].tolist())
    return X, L, names
