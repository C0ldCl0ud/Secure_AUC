import time

import crypten
import numpy as np
import pandas as pd
import torch
from crypten import cryptensor

@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment_approx(labels, predictions, thresholds):
    start = time.process_time()
    if len(predictions) != len(labels):
        raise Exception("Prediction and Reference have unequal length.")

    def encrypt(vector):
        # transform
        values = vector.iloc[:, 0].tolist()
        x = torch.tensor(values)
        x_enc = crypten.cryptensor(x)
        return x_enc

    def SEC_classifier(prediction, t):
        t = torch.tensor(t)
        t_enc = cryptensor(t)

        compare = prediction >= t_enc

        return compare

    def compute_AUC(fpr, tpr):
        sum = crypten.cryptensor(torch.tensor([0]))
        for i in range(1, len(tpr)):
            part1 = tpr[i]+tpr[i-1]
            part2 = fpr[i]-fpr[i-1]
            sum += (part1 * part2) / 2
        crypten.print("sum", sum.get_plain_text())

    def newton_raphson(x, a, b, num=5):

        for i in range(num):
            temp = x * b
            temp = 2 - temp
            x = x * temp

        return x * a

    labels_enc = encrypt(labels)
    predictions_enc = encrypt(predictions)

    TP = torch.tensor(np.zeros(len(thresholds)))
    TN = torch.tensor(np.zeros(len(thresholds)))
    FP = torch.tensor(np.zeros(len(thresholds)))
    FN = torch.tensor(np.zeros(len(thresholds)))
    TPR = torch.tensor(np.zeros(len(thresholds)))
    FPR = torch.tensor(np.zeros(len(thresholds)))

    TP = crypten.cryptensor(TP)
    TN = crypten.cryptensor(TN)
    FP = crypten.cryptensor(FP)
    FN = crypten.cryptensor(FN)
    TPR = crypten.cryptensor(TPR)
    FPR = crypten.cryptensor(FPR)

    nr_estimate = 1 / (len(labels_enc) / 2)

    for i in range(len(thresholds)):
        classifications = predictions_enc >= thresholds[i]
        TP_class = labels_enc * classifications
        TP[i] =  TP_class.sum()
        TN_class = (1-labels_enc) * (1-classifications)
        TN[i] = TN_class.sum()
        FP_class = (1 - labels_enc) * classifications #FP
        FP[i] = FP_class.sum()
        FN_class = labels_enc * (1 - classifications) #FN
        FN[i] = FN_class.sum()

        TPR[i] = newton_raphson(nr_estimate, TP[i], (TP[i] + FN[i]) )
        FPR[i] = newton_raphson(nr_estimate, FP[i], (FP[i] + TN[i]) )

    compute_AUC(FPR, TPR)

    end = time.process_time()

    # get the execution time
    time_overall = end - start

    print('Execution time:', time_overall, 'seconds')


@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment(data):
    start1 = time.process_time()
    def newton_raphson(x, a, b, num=5):

        for i in range(num):
            temp = x * b
            temp = 2 - temp
            x = x * temp

        return x * a

    def compute_AUC(fpr, tpr):
        sum = crypten.cryptensor(torch.tensor([0]))
        for i in range(1, len(tpr)):
            part1 = tpr[i]+tpr[i-1]
            part2 = fpr[i]-fpr[i-1]
            sum += (part1 * part2) / 2
        crypten.print("sum", sum.get_plain_text())

    def sortMergeJoin(left_label, left_pred, right_label, right_pred):
        predictions = torch.tensor(np.zeros(len(left_pred)+len(right_pred)))
        predictions = crypten.cryptensor(predictions)
        labels = torch.tensor(np.zeros(len(predictions)))
        labels = crypten.cryptensor(labels)

        left_index = 0
        right_index = 0
        counter = 0

        while left_index < len(left_pred) and right_index < len(right_pred):

            compare = left_pred[left_index] >= right_pred[right_index]
            predictions[counter] = compare * left_pred[left_index] + (1-compare) * right_pred[right_index]
            labels[counter] = compare * left_label[left_index] + (1-compare) * right_label[right_index]

            compare = compare.get_plain_text()
            left_index += compare
            right_index += (1 - compare)
            counter += 1

        return labels, predictions


    left = data[0]
    right = data[1]

    left_label = torch.tensor(left.iloc[:, 0].tolist())
    right_label = torch.tensor(right.iloc[:, 0].tolist())
    left_pred = torch.tensor(left.iloc[:, 1].tolist())
    right_pred = torch.tensor(right.iloc[:, 1].tolist())

    left_label = crypten.cryptensor(left_label)
    right_label = crypten.cryptensor(right_label)
    left_pred = crypten.cryptensor(left_pred)
    right_pred = crypten.cryptensor(right_pred)

    labels_enc, predictions_enc = sortMergeJoin(left_label, left_pred, right_label, right_pred)

    TP = torch.tensor(np.zeros(len(predictions_enc)))
    TN = torch.tensor(np.zeros(len(predictions_enc)))
    FP = torch.tensor(np.zeros(len(predictions_enc)))
    FN = torch.tensor(np.zeros(len(predictions_enc)))
    TPR = torch.tensor(np.zeros(len(predictions_enc)))
    FPR = torch.tensor(np.zeros(len(predictions_enc)))

    TP = crypten.cryptensor(TP)
    TN = crypten.cryptensor(TN)
    FP = crypten.cryptensor(FP)
    FN = crypten.cryptensor(FN)
    TPR = crypten.cryptensor(TPR)
    FPR = crypten.cryptensor(FPR)

    nr_estimate = 1/(len(labels_enc)/2)

    for i in range(len(predictions_enc)):
        classifications = predictions_enc >= predictions_enc[i]
        TP_class = labels_enc * classifications
        TP[i] =  TP_class.sum()
        TN_class = (1-labels_enc) * (1-classifications)
        TN[i] = TN_class.sum()
        FP_class = (1 - labels_enc) * classifications #FP
        FP[i] = FP_class.sum()
        FN_class = labels_enc * (1 - classifications) #FN
        FN[i] = FN_class.sum()

        TPR[i] = newton_raphson(nr_estimate, TP[i], (TP[i] + FN[i]) )
        FPR[i] = newton_raphson(nr_estimate, FP[i], (FP[i] + TN[i]) )


    compute_AUC(FPR, TPR)

    end1 = time.process_time()
    time_overall1 = end1 - start1

    print('Execution time:', time_overall1, 'seconds')


