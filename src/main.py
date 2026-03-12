"""
1) load data
2) preprocess data? encrypted?
3) encrypt data
4) test/train split
5) mpc traing
6) calculate mpc roc for thresholds
7) plot auc

where does dp fit in?
"""
import pandas as pd

import auc_analysis
import data_loader
import dp
import mpc
import numpy as np
import time

import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Compute thresholds for predictions")

    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="How many threshold values are computed (default: 100)"
    )

    parser.add_argument(
        "--partitions",
        type=int,
        default=1,
        help="Size / number of partitions for splitting the data (default: 1)"
    )

    parser.add_argument(
        "--label_path",
        type=str,
        default="",
        help="Path to label file (default: "")"
    )

    parser.add_argument(
        "--prediction_path",
        type=str,
        default="",
        help="Path to prediction file (default: "")"
    )

    return parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = ","
#device = torch.device("cpu")
#print(f"Using device: {device}")

paths_full = {
    "data/labels_100.txt": "data/pred_cons_100.txt",
    "data/labels_1000.txt": "data/pred_cons_1000.txt",
    "data/labels_10000.txt": "data/pred_cons_10000.txt",
    "data/labels_100000.txt": "data/pred_cons_100000.txt",
}
paths_demo = {
    "../data/labels_100.txt": "../data/pred_cons_100.txt"
}
epsilons = [0, 0.3, 1, 3, 9]

def calc_scikit_auc(data):
    labels, predictions = pd.DataFrame(data.iloc[:, 0]), pd.DataFrame(data.iloc[:, 1])
    auc_analysis.calculate_auc_scikit(labels, predictions)

def load_data(label_path, prediction_path):
    print(f"Lade Daten aus: {label_path}")
    labels = data_loader.load_data(label_path)

    print(f"Lade Daten aus: {prediction_path}")
    predictions = data_loader.load_data(prediction_path)

    data = data_loader.merge_df(labels, predictions)

    return data

def secure_auc(data, n_steps=100):
    thresholds = np.linspace(1, 0, n_steps)
    for epsilon in epsilons:
        TP = []
        FP = []
        P = []
        N = []
        for i in range(len(data)):
            tp, fp, p, n = mpc.count_stats(np.array(data[i].iloc[:,0]), np.array(data[i].iloc[:,1]), thresholds, epsilon)
            TP.append(tp)
            FP.append(fp)
            P.append(p)
            N.append(n)
        mpc.calculate_AUC(TP, FP, P, N, epsilon)

def dp_auc_calc(data, n_steps=1000):
    def sortMergeJoin(data):
        if len(data) == 1:
            return data[0][0], data[0][1]

        left = data.pop(0)
        right = data.pop(0)

        if isinstance(left, pd.DataFrame):
            left_label = left.iloc[:, 0].tolist()
            left_pred = left.iloc[:, 1].tolist()
        else:
            left_label = left[0]
            left_pred = left[1]

        if isinstance(right, pd.DataFrame):
            right_label = right.iloc[:, 0].tolist()
            right_pred = right.iloc[:, 1].tolist()
        else:
            right_label = right[0]
            right_pred = right[1]


        predictions = np.zeros(len(right_pred)+len(left_pred))
        labels = np.zeros(len(predictions))

        left_index = 0
        right_index = 0
        counter = 0

        while left_index < len(left_pred) and right_index < len(right_pred):

            if left_pred[left_index] >= right_pred[right_index]:
                predictions[counter] = left_pred[left_index]
                labels[counter] = left_label[left_index]
                left_index += 1
            else:
                predictions[counter] = right_pred[right_index]
                labels[counter] = right_label[right_index]
                right_index += 1

            counter += 1
        data.append((labels, predictions))
        return sortMergeJoin(data)



    start1 = time.process_time()

    TP = []
    FP = []
    P = []
    N = []

    thresholds = np.linspace(1, 0, n_steps)
    for epsilon in epsilons:
        for i in range(len(data)):

            labels, predictions = np.array(data[i].iloc[:,0]), np.array(data[i].iloc[:,1])

            tp = np.zeros(len(thresholds))
            fp = np.zeros(len(thresholds))

            p = labels.sum() + dp.laplace_noise(epsilon, 1)
            n = len(labels) - p

            for i in range(len(thresholds)):

                classifications = predictions >= thresholds[i]

                TP_class = labels * classifications
                tp[i] =  TP_class.sum() + dp.laplace_noise(epsilon, 1)

                FP_class = (1 - labels) * classifications
                fp[i] = FP_class.sum() + dp.laplace_noise(epsilon, 1)

            TP.append(tp)
            FP.append(fp)
            P.append(p)
            N.append(n)

        mpc.compute_AUC_without_mpc(TP, FP, P, N, epsilon)

    # get the execution time
    end1 = time.process_time()
    time_overall1 = end1 - start1
    print('Execution time:', time_overall1, 'seconds')


if __name__ == '__main__':
    args = parse_args()

    def run(labels_path, predictions_path):
        print("-------------------------------------------------------------------------")
        data = load_data(labels_path, predictions_path)
        print("Calculating scikit AUC:")
        calc_scikit_auc(data)

        data = data_loader.split_shuffled_df(data, 2)

        print("-------------------------------------------------------------------------")
        print("MPC-CALCULATIONS")
        print()

        print(
            f"Calculating approximate AUC (n_steps: {args.n_steps}):")
        secure_auc(data, n_steps=args.n_steps)

        print("-------------------------------------------------------------------------")
        print("DP-ONLY-CALCULATIONS")
        print()

        print("-------------------------------------------------------------------------")
        print(f"Calculating DP-only AUC:")
        dp_auc_calc(data)

    if len(args.label_path) != 0:
        run(labels_path=args.label_path, predictions_path=args.prediction_path)
    else:
        for labels_path, predictions_path in paths_full.items():
            run(labels_path, predictions_path)

