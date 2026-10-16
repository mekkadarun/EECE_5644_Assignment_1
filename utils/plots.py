# utils/plots.py

import matplotlib.pyplot as plt
#  Markers for the scatter plot
MARKERS = ["o", "s", "^"]
import numpy as np
from scipy.stats import norm


def plot_roc(fpr, tpr, i_best, i_theory, title, path):
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")     # ROC curve: one (FPR, TPR) point per threshold
    
    plt.plot(fpr[i_theory], tpr[i_theory], "go", markersize=10, fillstyle="none", label="theoretical gamma")     # Hollow green circle at the theoretical threshold
    plt.plot(fpr[i_best], tpr[i_best], "r*", markersize=15, label="min P(error)")     # Red star at the threshold with the smallest P(error)
    
    plt.xlabel("P(D=1|L=0) false positive")
    plt.ylabel("P(D=1|L=1) true positive")
    plt.title(title)

    plt.legend()
    
    plt.savefig(path, dpi=150)
    plt.close()
    
def plot_scatter(X, L, D, title, path):
    """3-D scatter: marker shape = true class, green = correct, red = wrong."""
   
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    
    # True whenever the decision was right
    correct = (D == L)
    
    # Loops over the true classes: ok means correct, - means not, bad means wrong
    for j in range(3):
        ok = (L == j) & correct
        bad = (L == j) & ~correct
    
        #  Draw the correct points on scatter plot
        ax.scatter(X[ok, 0], X[ok, 1], X[ok, 2], c="green", marker=MARKERS[j], s=10, alpha=0.3, label=f"class {j+1} correct")
        
        #  Draw the wrong points on scatter plot
        ax.scatter(X[bad, 0], X[bad, 1], X[bad, 2], c="red", marker=MARKERS[j], s=10, alpha=0.8, label=f"class {j+1} wrong")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    
    #  Rotate the view before saving
    ax.view_init(elev=20, azim=30)


    # Save the figure
    plt.savefig(path, dpi=150)
    plt.close()
    
def plot_alpha_sweep(alphas, errors, chosen, title, path):
    """Plot P(error) against alpha on a log axis, marking the alpha actually used."""
    plt.figure()
    plt.semilogx(alphas, errors, "o-")
    # Vertical line across the plot
    plt.axvline(chosen, color="r", linestyle="--", label=f"alpha used = {chosen}")
    plt.xlabel("alpha")
    plt.ylabel("P(error) on training samples")
    plt.title(title)
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_pca_2d(Z, y, class_names, explained, title, path):
    """Scatter of the first 2 principal components, one color per class."""
    plt.figure(figsize=(8, 6))
    for c in range(len(class_names)):
        pts = Z[y == c]
        plt.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.4, label=class_names[c])

    plt.xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    plt.ylabel(f"PC2 ({explained[1]:.1%} of variance)")
    plt.title(title)
    plt.legend(markerscale=4)
    plt.savefig(path, dpi=150)
    plt.close()

def plot_pca_3d(Z, y, class_names, explained, title, path):
    """Scatter of the first 3 principal components, one color per class."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    for c in range(len(class_names)):
        pts = Z[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, alpha=0.4, label=class_names[c])

    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_zlabel(f"PC3 ({explained[2]:.1%})")
    ax.set_title(title)
    ax.legend(markerscale=4, fontsize=8)
    plt.savefig(path, dpi=150)
    plt.close()

def plot_hist_vs_gaussian(x, title, path):
    """Histogram of one feature for one class, with the Gaussian fitted to it drawn on top."""
    plt.figure()
    plt.hist(x, bins=50, density=True, alpha=0.6, label="data")

    grid = np.linspace(x.min(), x.max(), 200)
    plt.plot(grid, norm.pdf(grid, x.mean(), x.std()), "r", label="fitted Gaussian")

    plt.title(title)
    plt.ylabel("density")
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.close()
