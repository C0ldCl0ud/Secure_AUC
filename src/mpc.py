import time

import crypten
import numpy as np
import torch

import dp


def count_stats(labels, predictions, thresholds, epsilon):
    TP = np.zeros(len(thresholds))
    FP = np.zeros(len(thresholds))
    P = labels.sum() +  + dp.laplace_noise(epsilon, sensitivity=1)
    N = len(labels) - P

    for i in range(len(thresholds)):
        classifications = predictions >= thresholds[i]

        TP_class = labels * classifications
        FP_class = (1 - labels) * classifications


        TP[i] = TP_class.sum() + dp.laplace_noise(epsilon, sensitivity=1)
        FP[i] = FP_class.sum() + dp.laplace_noise(epsilon, sensitivity=1)

    return TP, FP, P, N

def compute_AUC_without_mpc(tp_partial, fp_partial, P_partial, N_partial, epsilon):
    sum = 0

    tp = np.zeros(len(tp_partial[0]))
    fp = np.zeros(len(fp_partial[0]))
    P = 0
    N = 0

    for i in range(len(tp_partial)):
        tp += tp_partial[i]
        fp += fp_partial[i]
        P += P_partial[i]
        N += N_partial[i]

    for i in range(1, len(tp)):
        tpr = tp[i]+tp[i-1]
        fpr = fp[i]-fp[i-1]
        sum += tpr * fpr

    factor = 2*N*P
    auc = sum/factor
    print(f"AUC (epsilon: {epsilon}): ", auc)

@crypten.mpc.run_multiprocess(world_size=2)
def calculate_AUC(tp_partial, fp_partial, P_partial, N_partial, epsilon, scale_exponent):
    start = time.process_time()
    tp = crypten.cryptensor(torch.tensor(np.zeros(len(tp_partial[0]))))
    fp = crypten.cryptensor(torch.tensor(np.zeros(len(tp_partial[0]))))
    P = crypten.cryptensor(torch.tensor(0))
    N = crypten.cryptensor(torch.tensor(0))

    for i in range(len(tp_partial)):
        tp += crypten.cryptensor(torch.tensor(tp_partial[i]))
        fp += crypten.cryptensor(torch.tensor(fp_partial[i]))
        P += P_partial[i]
        N += N_partial[i]

    tpr = crypten.cryptensor(torch.tensor(np.zeros(len(tp))))
    fpr = crypten.cryptensor(torch.tensor(np.zeros(len(tp))))
    sum = crypten.cryptensor(torch.tensor([0]))
    for i in range(1, len(tp)):
        tpr[i] = tp[i] + tp[i - 1]
        fpr[i] = fp[i] - fp[i - 1]

    for i in range(scale_exponent):
        P *= 0.5
        N *= 0.5
        tpr *= 0.5
        fpr *= 0.5

    sum = tpr * fpr
    sum = sum.sum()
    auc = 0.5 * P.reciprocal()
    auc *= sum
    auc *= N.reciprocal()

    crypten.print(f"AUC (epsilon: {epsilon}): ", auc.get_plain_text())
    stop = time.process_time()
    crypten.print(f"Total time: {stop - start}")




