# utils/roc.py

import numpy as np

def roc_curve(score, L, priors):
    thresholds = np.concatenate(([-np.inf], np.sort(score), [np.inf])) # Build the list of thresholds
    
    #  Count the number of samples in each class
    n0 = np.sum(L==0)
    n1 = np.sum(L==1)
    
    #  Compare every score with every threshold at once, build a T/F table
    D = score[None, :] > thresholds[:, None]
    
    tpr = np.sum(D & (L==1), axis=1)/n1 # True positive rate
    fpr = np.sum(D & (L==0), axis=1)/n0 # False positive rate
    
    #  P(error) formula: (1 - tpr) = P(D=0|L=1)
    p_err = fpr*priors[0] + (1 - tpr) * priors[1]
    
    return thresholds, tpr, fpr, p_err