# q2.py

import os
import numpy as np
from utils.data import sample_gaussian_mixture, save_data, load_data
from utils.decide import class_likelihoods, posteriors, erm_decide, confusion_matrix
from utils.plots import plot_scatter

# ---- Problem setup: 4 Gaussian components, 3 classes ----
class_priors = [0.3, 0.3, 0.4]
# Weights of the four components
comp_weights = [0.3, 0.3, 0.2, 0.2] 

#  Means
# First two corners - x-axis, 2.5 apart
m1 = np.array([0.0, 0.0, 0.0])
m2 = np.array([2.5, 0.0, 0.0])
# Third corner - equilateral triangle on the floor 
m3 = np.array([1.25, 2.165, 0.0])
# Fourth corner - above the center of triangle
m4 = np.array([1.25, 0.722, 2.041])
#  List of means in component order
means = [m1, m2, m3, m4]

#  Covariances
I = np.eye(3)
covs = [I, I, I, I]

# Check the spacing
# Debugging: confirms every pair of means is 2.5 apart (commented out, not needed for results)
# for a in range(4):
#     for b in range(a + 1, 4):
#         print("Distance between mean", a + 1, "and", b + 1, ":", np.linalg.norm(means[a] - means[b]))

#  Pointing to the data file
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "q2_data.npz")

# Generate once, then reuse
if os.path.exists(DATA_PATH):
    X, comp = load_data(DATA_PATH)
    print("Loaded existing data from", DATA_PATH)
else:
    X, comp = sample_gaussian_mixture(comp_weights, means, covs, n_samples=10000, seed=7)
    save_data(DATA_PATH, X, comp)
    print("Generated and saved new data to", DATA_PATH)

#  Turn components into classes
L = np.minimum(comp, 2)
# Debugging: checks the data shape and that class fractions match the priors
# print("X shape:", X.shape)
# print("Class fractions:", np.bincount(L) / len(L))

#  Compute class likelihoods p(x|L=j)
lik = class_likelihoods(X, means, covs)
# Debugging: checks the likelihood table shape and a few sample values
# print("Likelihood table shape:", lik.shape)
# print("First 3 samples:\n", lik[:, :3].round(4))
# print("Their true classes:", L[:3])

#  Applying Bayes' rule to get posteriors
post = posteriors(lik, class_priors)
# Debugging: checks that posteriors look sensible and each column sums to 1
# print("Posteriors of first 3 samples:\n", post[:, :3].round(3))
# print("Column sums (should all be 1):", post.sum(axis=0)[:5])

# ---- Part A: MAP classification (0-1 loss) ----
loss_01 = 1 - np.eye(3)
D_map = erm_decide(post, loss_01)
# Debugging: checks a few decisions, and that ERM with 0-1 loss equals MAP (argmax of posteriors)
# print("Decisions for first 3 samples:", D_map[:3])
# print("Same as argmax of posteriors?", np.all(D_map == np.argmax(post, axis=0)))

counts_map, conf_map = confusion_matrix(D_map, L, 3)

print("Confusion counts, MAP:\n", counts_map.astype(int))
print("Confusion matrix P(D=i|L=j), MAP:\n", conf_map.round(3))
# Debugging: checks that each confusion-matrix column sums to 1
# print("Column sums (should be 1):", conf_map.sum(axis=0))
print("P(error), MAP:", np.mean(D_map != L))

FIGS = os.path.join(HERE, "figs")
plot_scatter(X, L, D_map, "Part A: MAP (0-1 loss)", os.path.join(FIGS, "q2_partA_scatter.png"))

# ---- Part B: ERM with loss matrices that penalize class-3 mistakes ----
#  Initializing given losses
loss_10 = np.array([[0, 1, 10],
                    [1, 0, 10],
                    [1, 1, 0]])

loss_100 = np.array([[0, 1, 100],
                     [1, 0, 100],
                     [1, 1, 0]])

D_10 = erm_decide(post, loss_10)
counts_10, conf_10 = confusion_matrix(D_10, L, 3)

# Estimated minimum risk loss_10
risk_10 = np.mean(loss_10[D_10, L])

print("Confusion matrix, Lambda10:\n", conf_10.round(3))
print("P(error), Lambda10:", np.mean(D_10 != L))
print("Expected risk, Lambda10:", risk_10)

#  Draw scatter plot for loss_10
plot_scatter(X, L, D_10, "Part B: ERM with Lambda10", os.path.join(FIGS, "q2_partB_L10_scatter.png"))

print("Expected risk, 0-1 loss:", np.mean(loss_01[D_map, L]))
print("Risk of MAP decisions under Lambda10:", np.mean(loss_10[D_map, L]))

# Repeat the steps for loss_100
D_100 = erm_decide(post, loss_100)
counts_100, conf_100 = confusion_matrix(D_100, L, 3)
risk_100 = np.mean(loss_100[D_100, L])

print("Confusion matrix, Lambda100:\n", conf_100.round(3))
print("P(error), Lambda100:", np.mean(D_100 != L))
print("Expected risk, Lambda100:", risk_100)
print("Risk of MAP decisions under Lambda100:", np.mean(loss_100[D_map, L]))

# Draw the scatter plot for loss_100
plot_scatter(X, L, D_100, "Part B: ERM with Lambda100", os.path.join(FIGS, "q2_partB_L100_scatter.png"))
