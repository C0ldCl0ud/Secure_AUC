import numpy as np
import auc_analysis
import mpc

def differential_auc():
    sensitivity = 1

    epsilon = 0.1
    scale = sensitivity / epsilon

    noise = np.random.laplace(loc=0, scale=scale)

    noisy_auc_score_scikit =  auc_analysis.auc + noise

    noisy_auc_score = mpc. + noise