# q1.py

import os
import numpy as np

from utils.data import sample_gaussian_mixture, save_data, load_data
from utils.gaussian import log_gaussian
from utils.roc import roc_curve
from utils.plots import plot_roc
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# ---- Problem setup: class priors, means and covariances ----
priors = [0.35, 0.65]

m0 = np.array([-0.5, -0.5, -0.5, -0.5])
C0 = 0.25 * np.array([[ 2.0, -0.5,  0.3, 0.0],
                      [-0.5,  1.0, -0.5, 0.0],
                      [ 0.3, -0.5,  1.0, 0.0],
                      [ 0.0,  0.0,  0.0, 2.0]])

m1 = np.array([1.0, 1.0, 1.0, 1.0])
C1 = np.array([[ 1.0, 0.3, -0.2, 0.0],
               [ 0.3, 2.0,  0.3, 0.0],
               [-0.2, 0.3,  1.0, 0.0],
               [ 0.0, 0.0,  0.0, 3.0]])

means = [m0, m1]
covs = [C0, C1]

# Generate the 10K samples once, then always reuse the saved file
DATA_PATH = "data/q1_data.npz"

if os.path.exists(DATA_PATH):
    X, L = load_data(DATA_PATH)
    print(f"Loaded existing data from {DATA_PATH}")
else:
    X, L = sample_gaussian_mixture(priors, means, covs, n_samples=10000, seed=42)
    save_data(DATA_PATH, X, L)
    print(f"Generated and saved new data to {DATA_PATH}")

# Sanity checks
print("X shape:", X.shape)
print("Fraction of class-1 samples:", L.mean())
print("Sample mean of class 0:", X[L == 0].mean(axis=0).round(2))
print("Sample mean of class 1:", X[L == 1].mean(axis=0).round(2))

# ---- Part A: ERM classifier with the true pdfs ----
score = log_gaussian(X, m1, C1) - log_gaussian(X, m0, C0) # Compute the log-likelihood ratio for all 10000 samples
gamma_theory = priors[0] / priors[1] # theoretical threshold

# Decide D=1 when the score is above log(gamma)
D = (score > np.log(gamma_theory)).astype(int)
# Fraction of samples whose decision differs from the true label
print("P(error) at theoretical gamma:", np.mean(D != L))
# Sweep all thresholds to get the ROC curve and P(error) at each one
thr, tpr, fpr, perr = roc_curve(score, L, priors)

i = np.argmin(perr) # Index of the threshold with the smallest P(error)
i_theory = np.argmin(np.abs(thr - np.log(gamma_theory))) # Index of the sweep threshold closest to the theoretical log(gamma)

print("Empirical gamma:", np.exp(thr[i]))
print("Min P(error):", perr[i])
print("At that point TPR =", tpr[i], "FPR =", fpr[i])
print("P(error) at theoretical gamma (from ROC):", perr[i_theory])
print("At theoretical gamma: TPR =", tpr[i_theory], "FPR =", fpr[i_theory])

plot_roc(fpr, tpr, i, i_theory, "Part A: ROC with true pdf", "figs/q1_partA_roc.png")

# ---- Part B: Naive Bayes (diagonal covariances, true means) ----

# Naive Bayes assumption: diagonal entries are the 4 variances and zeroes everywhere else
C0_nb = np.diag(np.diag(C0)) 
C1_nb = np.diag(np.diag(C1))
print("Naive Bayes C0:\n", C0_nb)

#  Log-likelihood ratio with true means
score_nb = log_gaussian(X, m1, C1_nb) - log_gaussian(X, m0, C0_nb)

thr_nb, tpr_nb, fpr_nb, perr_nb = roc_curve(score_nb, L, priors)

i_nb = np.argmin(perr_nb) # Index of the threshold with the smallest P(error) under Naive Bayes

i_theory_nb = np.argmin(np.abs(thr_nb - np.log(gamma_theory))) # INdex of the threshold closest to theoretical ln gamma

print("NB empirical gamma:", np.exp(thr_nb[i_nb]))
print("NB min P(error):", perr_nb[i_nb])
print("NB at min: TPR =", tpr_nb[i_nb], "FPR =", fpr_nb[i_nb])
print("NB P(error) at theoretical gamma:", perr_nb[i_theory_nb])
print("NB at theoretical gamma: TPR =", tpr_nb[i_theory_nb], "FPR =", fpr_nb[i_theory_nb])

plot_roc(fpr_nb, tpr_nb, i_nb, i_theory_nb, "Part B: ROC with Naive Bayes", "figs/q1_partB_roc.png")

plt.figure()
plt.plot(fpr, tpr, label="Part A: true pdf")
plt.plot(fpr_nb, tpr_nb, "--", label="Part B: Naive Bayes")

plt.xlabel("P(D=1|L=0)")
plt.ylabel("P(D=1|L=1)")
plt.legend()
plt.savefig("figs/q1_compare_AB.png", dpi=150)
plt.close()

# ---- Part C: Fisher LDA (estimated from samples) ----
# Estimated means
mu0_hat = X[L == 0].mean(axis=0)
mu1_hat = X[L == 1].mean(axis=0)
#  Estimated covariances
S0_hat = np.cov(X[L == 0], rowvar=False)
S1_hat = np.cov(X[L == 1], rowvar=False)

# Vector pointing from the class-0 mean to the class-1 mean
diff = mu1_hat - mu0_hat

# Within-class scatter with equal weights
Sb = np.outer(diff, diff)

Sw = S0_hat + S1_hat

#  Solve for eigen value to get the Fisher ratio
vals, vecs = eigh(Sb, Sw)

w_lda = vecs[:, np.argmax(vals)] # Pick the column with largest eigenvalue

proj = X @ w_lda

# If class 1 projects lower on average, flip both
if proj[L == 1].mean() < proj[L == 0].mean():
    w_lda = -w_lda
    proj = -proj

print("w_LDA:", w_lda)

thr_lda, tpr_lda, fpr_lda, perr_lda = roc_curve(proj, L, priors)
i_lda = np.argmin(perr_lda)

print("LDA best tau:", thr_lda[i_lda])
print("LDA min P(error):", perr_lda[i_lda])
print("LDA at min: TPR =", tpr_lda[i_lda], "FPR =", fpr_lda[i_lda])

plt.figure()
plt.plot(fpr_lda, tpr_lda, label="ROC (Fisher LDA)")
plt.plot(fpr_lda[i_lda], tpr_lda[i_lda], "r*", markersize=15, label="min P(error)")
plt.xlabel("P(D=1|L=0) false positive")
plt.ylabel("P(D=1|L=1) true positive")
plt.title("Part C: ROC with Fisher LDA")
plt.legend()
plt.savefig("figs/q1_partC_roc.png", dpi=150)
plt.close()

# Compare all three classifiers on one zoomed ROC plot
plt.figure()
plt.plot(fpr, tpr, label="A: true pdf")
plt.plot(fpr_nb, tpr_nb, "--", label="B: Naive Bayes")
plt.plot(fpr_lda, tpr_lda, ":", label="C: Fisher LDA")
plt.xlabel("P(D=1|L=0) false positive")
plt.ylabel("P(D=1|L=1) true positive")
plt.title("ROC comparison: true pdf vs Naive Bayes vs LDA")
plt.legend()
plt.savefig("figs/q1_compare_all.png", dpi=150)
plt.close()
