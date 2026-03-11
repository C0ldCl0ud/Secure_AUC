import numpy as np
import auc_analysis
import mpc

def laplace_noise(epsilon, sensitivity):
    if epsilon == 0:
        return 0
    return np.random.laplace(loc=0, scale=sensitivity/epsilon)

def differential_auc_scilearn(truth, prediction):

    auc = auc_analysis.calculate_auc_scikit(truth, prediction)

    sensitivity = 1

    epsilon = 0.1
    scale = sensitivity / epsilon

    noise = np.random.laplace(loc=0, scale=scale)

    noisy_auc_score_scikit =  auc + noise


def differential_auc_approx(labels, predictions, thresholds):
    approx = mpc.run_experiment_approx(labels, predictions, thresholds)

    #Unwrap MPC return value: list -> float
    if isinstance(approx, list):
        approx = approx[0]
    if hasattr(approx, "item"):
        approx = approx.item()

    approx_auc = float(approx)

    #da fläche aus FN/TP etc.
    sensitivity = 2/len(labels)
    epsilon = 9
    scale = sensitivity / epsilon

    noise1 = np.random.laplace(loc=0, scale=scale)

    noisy_auc_score_mpc = approx_auc + noise1

    print(f"With noise: {noisy_auc_score_mpc}")




#def differential_auc_approx(labels, predictions, thresholds):
 #   approx_auc = mpc.run_experiment_approx(labels, predictions, thresholds)

  #  sensitivity = 1

   # epsilon = 0.1
    #scale = sensitivity / epsilon

    #noise1 = np.random.laplace(loc=0, scale=scale)

    #noisy_auc_score_mpc = approx_auc + noise1

    #print(f"With noise: {noisy_auc_score_mpc}")

# in: def compute_AUC(fpr, tpr):
#auc_plain = sum.get_plain_text().item()
#return auc_plain