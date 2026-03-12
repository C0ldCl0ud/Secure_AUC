import time

import crypten
import numpy as np
import torch
import pandas as pd

import dp


@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment_real(data, epsilons):

    def compute_AUC(tp, fp, P, N):
        for n in range(len(tp)):
            sum = crypten.cryptensor(torch.tensor([0]))
            for i in range(1, len(tp[n])):
                tpr = tp[n][i]+tp[n][i-1]
                fpr = fp[n][i]-fp[n][i-1]
                sum += tpr * fpr

            scale = crypten.cryptensor(torch.tensor([1]))
            for i in range(6):
                P[n] *= 0.5
                N[n] *= 0.5
                scale *= 2

            auc = 0.5 * P[n].reciprocal() * N[n].reciprocal()  * sum * scale.reciprocal() * scale.reciprocal()

            crypten.print(f"AUC (epsilon: {epsilons[n]}): ", auc.get_plain_text())

    def sortMergeJoin(data):
        if len(data) == 1:
            return data[0][0], data[0][1]

        left = data.pop(0)
        right = data.pop(0)

        if isinstance(left, pd.DataFrame):
            left_label = crypten.cryptensor(torch.tensor(left.iloc[:, 0].tolist()))
            left_pred = crypten.cryptensor(torch.tensor(left.iloc[:, 1].tolist()))
        else:
            left_label = left[0]
            left_pred = left[1]

        if isinstance(right, pd.DataFrame):
            right_label = crypten.cryptensor(torch.tensor(right.iloc[:, 0].tolist()))
            right_pred = crypten.cryptensor(torch.tensor(right.iloc[:, 1].tolist()))
        else:
            right_label = right[0]
            right_pred = right[1]

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

        data.append((labels, predictions))
        return sortMergeJoin(data)

    start1 = time.process_time()

    start_sort = time.process_time()
    labels_enc, predictions_enc = sortMergeJoin(data)
    end_sort = time.process_time()
    crypten.print(f"Sorting finished after {(end_sort-start_sort)/60} minutes.")

    TP_noisy = []
    FP_noisy = []

    P = labels_enc.sum()
    P_noisy = []
    N_noisy = []
    for epsilon in epsilons:
        P_noisy.append(P + dp.laplace_noise(epsilon, 1))
        N_noisy.append(len(labels_enc) - P_noisy[-1])
        TP_noisy.append(crypten.cryptensor(torch.tensor(np.zeros(len(predictions_enc)))))
        FP_noisy.append(crypten.cryptensor(torch.tensor(np.zeros(len(predictions_enc)))))

    thresholds = len(predictions_enc)
    loop_start = time.time()
    for i in range(thresholds):
        if i % 1000 == 0 and i != 0:
            current_time = time.time()
            time_per_loop = (current_time - loop_start) / i
            crypten.print(f"{i}/{thresholds} iterations.")
            crypten.print(f"approximately {((thresholds-i)*time_per_loop)/60} minutes remaining.")

        classifications = predictions_enc >= predictions_enc[i]

        TP_class = labels_enc * classifications
        FP_class = (1 - labels_enc) * classifications

        for n in range(len(epsilons)):
            TP_noisy[n][i] = TP_class.sum() + dp.laplace_noise(epsilons[n], 1)
            FP_noisy[n][i] = FP_class.sum() + dp.laplace_noise(epsilons[n], 1)

    compute_AUC(TP_noisy, FP_noisy, P_noisy, N_noisy)

    # get the execution time
    end1 = time.process_time()
    time_overall1 = end1 - start1
    crypten.print('Execution time:', time_overall1, 'seconds')



@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment_approx(data, thresholds, epsilons, partitions=1):
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
        auc = []
        for n in range(len(tp)):
            sum = crypten.cryptensor(torch.tensor([0]))
            for i in range(1, len(tp[n])):
                tpr = tp[n][i]+tp[n][i-1]
                fpr = fp[n][i]-fp[n][i-1]
                sum += tpr * fpr

            scale = crypten.cryptensor(torch.tensor([1]))
            for i in range(6):
                P[n] *= 0.5
                N[n] *= 0.5
                scale *= 2

            auc.append(0.5 * P[n].reciprocal() * N[n].reciprocal() * sum * scale.reciprocal() * scale.reciprocal() * (1/partitions))
        return auc



    labels_enc = encrypt_df(labels)
    predictions_enc = encrypt_df(predictions)

    partial_auc = []
    stepwidth = int(len(labels_enc)/partitions)

    loop_start = time.time()
    for i in range(partitions):

        if i != 0:
            current_time = time.time()
            time_per_loop = (current_time - loop_start) / i
            crypten.print(f"{i}/{partitions} iterations.")
            crypten.print(f"approximately {((partitions-i)*time_per_loop)/60} minutes remaining.")

        P = labels_enc[i*stepwidth:(i+1)*stepwidth].sum()
        N = stepwidth - P

        TP_noisy = []
        FP_noisy = []
        P_noisy = []
        N_noisy = []
        for epsilon in epsilons:
            P_noisy.append(P + dp.laplace_noise(epsilon, 1))
            N_noisy.append(stepwidth - P_noisy[-1])
            TP_noisy.append(crypten.cryptensor(torch.tensor(np.zeros(len(thresholds)))))
            FP_noisy.append(crypten.cryptensor(torch.tensor(np.zeros(len(thresholds)))))

        for j in range(len(thresholds)):
            classifications = predictions_enc[i*stepwidth:(i+1)*stepwidth] >= thresholds[j]

            TP_class = labels_enc[i*stepwidth:(i+1)*stepwidth] * classifications
            FP_class = (1 - labels_enc[i*stepwidth:(i+1)*stepwidth]) * classifications

            for n in range(len(epsilons)):
                TP_noisy[n][j] = TP_class.sum() + dp.laplace_noise(epsilons[n], 1)
                FP_noisy[n][j] = FP_class.sum() + dp.laplace_noise(epsilons[n], 1)

        auc = compute_AUC(TP_noisy, FP_noisy, P_noisy, N_noisy)
        partial_auc.append(auc)

    sum = crypten.cryptensor(torch.tensor(np.zeros(len(epsilons))))
    for j in range(partitions):
        for k in range(len(epsilons)):
            sum[k] += partial_auc[j][k][0]

    for k in range(len(epsilons)):
        crypten.print(f'AUC (epsilon: {epsilons[k]}):', sum[k].get_plain_text())

     #get the execution time
    end = time.process_time()
    time_overall = end - start
    crypten.print('Execution time:', time_overall, 'seconds')

