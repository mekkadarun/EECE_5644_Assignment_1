# q3.py
import os
import numpy as np
from utils.data import load_wine, load_har
from utils.gaussian import fit_gaussian_classes, log_gaussian
from utils.decide import map_decide_log, confusion_matrix
import pandas as pd
from utils.plots import plot_alpha_sweep, plot_pca_2d, plot_pca_3d, plot_hist_vs_gaussian
from utils.pca import pca
from scipy.stats import skew


def report_conditioning(name, classes, covs, lams):
    """Print rank and eigenvalues of each class covariance before and after adding lambda*I."""
    #  d is the number of features
    d = covs[0].shape[0]
    print(f"--- {name}: covariance conditioning ---")
    # Loop over classes and recover the raw covariance
    for c in range(len(covs)):
        raw = covs[c] - lams[c] * np.eye(d)
        e_raw = np.linalg.eigvalsh(raw)
        e_reg = np.linalg.eigvalsh(covs[c])
        print(f"class {classes[c]}: rank {np.linalg.matrix_rank(raw)}/{d}, "
              f"raw eig min {e_raw.min():.1e} max {e_raw.max():.1e}")
        print(f"    after reg: min eig {e_reg.min():.2e} (lambda {lams[c]:.2e}), "
              f"condition number {e_reg.max() / e_reg.min():.1e}")
    
def run_variant(name, X, y, n_classes, shared):
    """Fit, classify and score one variant. Prints P(error), returns decisions and confusion matrices."""
    priors, means, covs, lams = fit_gaussian_classes(X, y, ALPHA, shared=shared)
    D = map_decide_log(X, priors, means, covs)
    counts, conf = confusion_matrix(D, y, n_classes)
    print(f"{name}: P(error) = {np.mean(D != y):.4f}")
    return D, counts, conf


HERE = os.path.dirname(os.path.abspath(__file__))
WINE_PATH = os.path.join(HERE, "data", "wine+quality", "winequality-white.csv")
HAR_DIR = os.path.join(HERE, "data", "UCI HAR Dataset", "UCI HAR Dataset")

X_w, L_w, wine_names = load_wine(WINE_PATH)
# Debugging: confirms the wine file loaded with the expected shape and class counts
# print("Wine:", X_w.shape, "classes and counts:", np.unique(L_w, return_counts=True))

# Standardized copy: every feature rescaled to mean 0, variance 1
X_ws = (X_w - X_w.mean(axis=0)) / X_w.std(axis=0)

X_h, L_h, har_names = load_har(HAR_DIR)
# Debugging: confirms train + test stacked to 10299 samples, and the activity names loaded
# print("HAR:", X_h.shape, "classes and counts:", np.unique(L_h, return_counts=True))
# print("Activities:", har_names)

# ---- Map labels to indices 0..C-1 ----
classes_w, y_w = np.unique(L_w, return_inverse=True)
classes_h, y_h = np.unique(L_h, return_inverse=True)

# ---- Fit class Gaussians ----
ALPHA = 0.05
priors_w, means_w, covs_w, lams_w = fit_gaussian_classes(X_w, y_w, ALPHA)
print("Wine priors:", np.round(priors_w, 4))
print("Wine lambdas:", np.round(lams_w, 3))
priors_h, means_h, covs_h, lams_h = fit_gaussian_classes(X_h, y_h, ALPHA)
print("HAR priors:", np.round(priors_h, 4))
print("HAR lambdas:", np.round(lams_h, 4))

report_conditioning("Wine", classes_w, covs_w, lams_w)
report_conditioning("HAR", classes_h, covs_h, lams_h)

raw_h0 = covs_h[0] - lams_h[0] * np.eye(X_h.shape[1])
try:
    log_gaussian(X_h[:5], means_h[0], raw_h0)
    print("Unregularized HAR covariance worked")
except Exception as err:
    print("Unregularized HAR covariance fails:", err)

# ---- MAP classification on all samples ----
D_w = map_decide_log(X_w, priors_w, means_w, covs_w)
counts_w, conf_w = confusion_matrix(D_w, y_w, len(classes_w))

print("Wine P(error):", np.mean(D_w != y_w))
print("Wine confusion counts (rows = decision, cols = true), order", classes_w, ":\n", counts_w.astype(int))
print("Wine P(D=i|L=j):\n", conf_w.round(3))

D_h = map_decide_log(X_h, priors_h, means_h, covs_h)
counts_h, conf_h = confusion_matrix(D_h, y_h, len(classes_h))
print("HAR P(error):", np.mean(D_h != y_h))
print("HAR confusion counts:\n", counts_h.astype(int))
print("HAR P(D=i|L=j):\n", conf_h.round(3))

# ---- Regularization variants ----
run_variant("Wine raw, per-class lambda", X_w, y_w, len(classes_w), False)
run_variant("Wine raw, shared lambda", X_w, y_w, len(classes_w), True)
run_variant("Wine standardized, per-class lambda", X_ws, y_w, len(classes_w), False)
D_wm, counts_wm, conf_wm = run_variant("Wine standardized, shared lambda", X_ws, y_w, len(classes_w), True)
run_variant("HAR per-class lambda", X_h, y_h, len(classes_h), False)
D_hm, counts_hm, conf_hm = run_variant("HAR shared lambda", X_h, y_h, len(classes_h), True)

wine_rows = [f"D={c}" for c in classes_w]
wine_cols = [f"L={c}" for c in classes_w]
print("Wine (main) confusion counts:\n", pd.DataFrame(counts_wm.astype(int), index=wine_rows, columns=wine_cols))
print("Wine (main) P(D=i|L=j):\n", pd.DataFrame(conf_wm.round(3), index=wine_rows, columns=wine_cols))

har_short = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY"]

print("HAR (main) confusion counts:\n", pd.DataFrame(counts_hm.astype(int), index=har_short, columns=har_short))
print("HAR (main) P(D=i|L=j):\n", pd.DataFrame(conf_hm.round(3), index=har_short, columns=har_short))

# ---- Alpha sensitivity check (alpha was fixed at 0.05 beforehand) ----
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5]
err_w, err_h = [], []
for a in ALPHAS:
    p, m, c, _ = fit_gaussian_classes(X_ws, y_w, a, shared=True)
    err_w.append(np.mean(map_decide_log(X_ws, p, m, c) != y_w))
    p, m, c, _ = fit_gaussian_classes(X_h, y_h, a, shared=True)
    err_h.append(np.mean(map_decide_log(X_h, p, m, c) != y_h))
    print(f"alpha {a}: wine {err_w[-1]:.4f}, HAR {err_h[-1]:.4f}")

FIGS = os.path.join(HERE, "figs")
plot_alpha_sweep(ALPHAS, err_w, ALPHA, "Wine: sensitivity to alpha", os.path.join(FIGS, "q3_wine_alpha.png"))
plot_alpha_sweep(ALPHAS, err_h, ALPHA, "HAR: sensitivity to alpha", os.path.join(FIGS, "q3_har_alpha.png"))

# ---- PCA visualizations ----
Z_w, expl_w = pca(X_ws)
Z_h, expl_h = pca(X_h)

print("Wine: first 3 PCs explain", expl_w[:3].round(3), "total", expl_w[:3].sum().round(3))
print("HAR: first 3 PCs explain", expl_h[:3].round(3), "total", expl_h[:3].sum().round(3))

wine_labels = [f"quality {c}" for c in classes_w]

plot_pca_2d(Z_w, y_w, wine_labels, expl_w, "Wine (standardized): first 2 PCs", os.path.join(FIGS, "q3_wine_pca2d.png"))
plot_pca_3d(Z_w, y_w, wine_labels, expl_w, "Wine (standardized): first 3 PCs", os.path.join(FIGS, "q3_wine_pca3d.png"))
plot_pca_2d(Z_h, y_h, har_names, expl_h, "HAR: first 2 PCs", os.path.join(FIGS, "q3_har_pca2d.png"))
plot_pca_3d(Z_h, y_h, har_names, expl_h, "HAR: first 3 PCs", os.path.join(FIGS, "q3_har_pca3d.png"))

# ---- Gaussianity check: skewness (0 for a Gaussian) and histograms ----
print("Wine quality-6 skewness per feature:\n", pd.Series(skew(X_w[y_w == 3], axis=0), index=wine_names).round(2))

s_sit = np.abs(skew(X_h[y_h == 3], axis=0))
print("HAR sitting: median |skew|", np.median(s_sit).round(2), ", fraction of features with |skew| > 1:", np.mean(s_sit > 1).round(2))

for i in [3, 4, 10]:
    plot_hist_vs_gaussian(X_w[y_w == 3, i], f"Wine quality 6: {wine_names[i]}", os.path.join(FIGS, f"q3_wine_hist_{i}.png"))

for i in [0, 200, 558]:
    plot_hist_vs_gaussian(X_h[y_h == 3, i], f"HAR sitting: feature {i + 1}", os.path.join(FIGS, f"q3_har_hist_{i}.png"))
