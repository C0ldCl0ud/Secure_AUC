import time

import crypten
import numpy as np
import torch
import pandas as pd


@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment_real(data):

    def compute_AUC(tp, fp, P, N):
        sum = crypten.cryptensor(torch.tensor([0]))
        for i in range(1, len(tp)):
            tpr = tp[i]+tp[i-1]
            fpr = fp[i]-fp[i-1]
            sum += tpr * fpr

        scale = crypten.cryptensor(torch.tensor([1]))
        for i in range(6):
            P *= 0.5
            N *= 0.5
            scale *= 2

        auc = 0.5 * P.reciprocal() * N.reciprocal()  * sum * scale.reciprocal() * scale.reciprocal()

        crypten.print("AUC: ", auc.get_plain_text())

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

    start1 = time.process_time()

    left = data[0]
    right = data[1]

    left_label = crypten.cryptensor(torch.tensor(left.iloc[:, 0].tolist()))
    right_label = crypten.cryptensor(torch.tensor(right.iloc[:, 0].tolist()))
    left_pred = crypten.cryptensor(torch.tensor(left.iloc[:, 1].tolist()))
    right_pred = crypten.cryptensor(torch.tensor(right.iloc[:, 1].tolist()))

    start_sort = time.process_time()
    labels_enc, predictions_enc = sortMergeJoin(left_label, left_pred, right_label, right_pred)
    end_sort = time.process_time()
    print(f"Sorting finished after {(end_sort-start_sort)/60} minutes.")

    TP = crypten.cryptensor(torch.tensor(np.zeros(len(predictions_enc))))
    FP = crypten.cryptensor(torch.tensor(np.zeros(len(predictions_enc))))

    P = labels_enc.sum()
    N = len(labels_enc) - P

    for i in range(len(predictions_enc)):
        classifications = predictions_enc >= predictions_enc[i]

        TP_class = labels_enc * classifications
        TP[i] =  TP_class.sum()

        FP_class = (1 - labels_enc) * classifications #FP
        FP[i] = FP_class.sum()

    compute_AUC(TP, FP, P, N)

    # get the execution time
    end1 = time.process_time()
    time_overall1 = end1 - start1
    crypten.print('Execution time:', time_overall1, 'seconds')



@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment_approx(data, thresholds, partitions=1):
    start = time.process_time()

    labels, predictions = pd.DataFrame(data.iloc[:, 0]), pd.DataFrame(data.iloc[:, 1])

    if len(predictions) != len(labels):
        raise Exception("Prediction and Reference have unequal length.")

    def encrypt_df(vector):
        # transform
        values = vector.iloc[:, 0].tolist()
        x = torch.tensor(values)
        x_enc = crypten.cryptensor(x)
        return x_enc

    def compute_AUC(tp, fp, P, N):
        sum = crypten.cryptensor(torch.tensor([0]))
        for i in range(1, len(tp)):
            tpr = tp[i] + tp[i - 1]
            fpr = fp[i] - fp[i - 1]
            sum += tpr * fpr

        scale = crypten.cryptensor(torch.tensor([1]))
        for i in range(6):
            P *= 0.5
            N *= 0.5
            scale *= 2

        return 0.5 * P.reciprocal() * N.reciprocal() * sum * scale.reciprocal() * scale.reciprocal()



    labels_enc = encrypt_df(labels)
    predictions_enc = encrypt_df(predictions)

    TP = crypten.cryptensor(torch.tensor(np.zeros(len(thresholds))))
    FP = crypten.cryptensor(torch.tensor(np.zeros(len(thresholds))))

    P = labels_enc.sum()
    N = len(labels_enc) - P

    partial_auc = crypten.cryptensor(torch.tensor([0]))
    stepwidth = int(len(labels_enc)/partitions)

    for i in range(partitions):
        for j in range(len(thresholds)):
            classifications = predictions_enc[i*stepwidth:(i+1)*stepwidth] >= thresholds[j]

            TP_class = labels_enc[i*stepwidth:(i+1)*stepwidth] * classifications
            TP[j] = TP_class.sum()

            FP_class = (1 - labels_enc[i*stepwidth:(i+1)*stepwidth]) * classifications
            FP[j] = FP_class.sum()

        partial_auc += compute_AUC(TP, FP, P, N)

    auc = partial_auc.get_plain_text()
    auc = auc / partitions
    crypten.print('AUC:', auc)

     #get the execution time
    end = time.process_time()
    time_overall = end - start
    crypten.print('Execution time:', time_overall, 'seconds')

