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

    parser.add_argument(
        "--skip_real",
        type=bool,
        default=False,
        help="Whether to skip real data, with long computation time (default: False)"
    )
    parser.add_argument(
        "--skip_approx",
        type=bool,
        default=False,
        help="Whether to skip approx data (default: False)"
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
    "data/labels_100.txt": "data/pred_cons_100.txt"
}

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

def secure_auc(data, epsilon=0, approx=False, n_steps=1000, partitions=1):
    if approx:
        data = data.sample(frac=1, random_state=42)
        thresholds = np.linspace(1, 0, n_steps)
        mpc.run_experiment_approx(data, thresholds, epsilon, partitions)
    else:
        data = data_loader.split_shuffled_df(data, 2)
        mpc.run_experiment_real(data, epsilon)

def dp_auc_calc(data, epsilon=0):

    def compute_AUC(tp, fp, P, N):
        sum = 0
        for i in range(1, len(tp)):
            tpr = tp[i]+tp[i-1]
            fpr = fp[i]-fp[i-1]
            sum += tpr * fpr
        factor = 2*N*P
        auc = sum/factor
        print("AUC: ", auc)

    def sortMergeJoin(left_label, left_pred, right_label, right_pred):
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

        return labels, predictions

    data = data_loader.split_shuffled_df(data, 2)

    start1 = time.process_time()

    left = data[0]
    right = data[1]

    left_label = left.iloc[:, 0].tolist()
    right_label = right.iloc[:, 0].tolist()
    left_pred = left.iloc[:, 1].tolist()
    right_pred = right.iloc[:, 1].tolist()

    start_sort = time.process_time()
    labels, predictions = sortMergeJoin(left_label, left_pred, right_label, right_pred)
    end_sort = time.process_time()
    print(f"Sorting finished after {(end_sort-start_sort)/60} minutes.")

    TP = np.zeros(len(predictions))
    FP = np.zeros(len(predictions))

    P = labels.sum() + dp.laplace_noise(epsilon, 1)
    N = len(labels) - P

    for i in range(len(predictions)):
        classifications = predictions >= predictions[i]

        TP_class = labels * classifications
        TP[i] =  TP_class.sum() + dp.laplace_noise(epsilon, 1)

        FP_class = (1 - labels) * classifications #FP
        FP[i] = FP_class.sum() +dp.laplace_noise(epsilon, 1)

    compute_AUC(TP, FP, P, N)

    # get the execution time
    end1 = time.process_time()
    time_overall1 = end1 - start1
    print('Execution time:', time_overall1, 'seconds')


if __name__ == '__main__':

    epsilons = [0, 0.3, 1, 3, 9]

    args = parse_args()


    def run(labels_path, predictions_path):
        print("-------------------------------------------------------------------------")
        data = load_data(labels_path, predictions_path)
        print("Calculating scikit AUC:")
        calc_scikit_auc(data)

        print("-------------------------------------------------------------------------")
        print("MPC-CALCULATIONS")
        print()

        for epsilon in epsilons:
            if not args.skip_real:
                print("-------------------------------------------------------------------------")
                print(f"Calculating accurate AUC(epsilon: {epsilon}):")
                secure_auc(data, epsilon=epsilon)
            if not args.skip_approx:
                print("-------------------------------------------------------------------------")
                print(
                    f"Calculating approximate AUC (epsilon: {epsilon}, n_steps: {args.n_steps}, partitions: {args.partitions}):")
                secure_auc(data, epsilon=epsilon, approx=True, n_steps=args.n_steps, partitions=args.partitions)

        print("-------------------------------------------------------------------------")
        print("DP-ONLY-CALCULATIONS")
        print()
        for epsilon in epsilons:
            print("-------------------------------------------------------------------------")
            print(f"Calculating DP-only AUC(epsilon: {epsilon}):")
            dp_auc_calc(data, epsilon=epsilon)

    if len(args.label_path) != 0:
        run(labels_path=args.label_path, predictions_path=args.prediction_path)
    else:
        print("hallo")
        for labels_path, predictions_path in paths_full.items():
            run(labels_path, predictions_path)

