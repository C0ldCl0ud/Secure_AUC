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
import math

import pandas as pd
from matplotlib.patches import Patch

import auc_analysis
import data_loader
import dp
import mpc
import numpy as np
import time
import matplotlib.pyplot as plt

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

paths = [
    ["../data/labels_100.txt", "../data/pred_cons_100.txt"],
    ["../data/labels_1000.txt", "../data/pred_cons_1000.txt"],
    ["../data/labels_10000.txt", "../data/pred_cons_10000.txt"],
    ["../data/labels_100000.txt", "../data/pred_cons_100000.txt"]
]
dataset = [100, 1000, 10000, 100000]

epsilons = (np.linspace(0,1, 20)**2)*10

#steps = [8,32,128]
steps = [2,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128]


def calc_scikit_auc(data):
    labels, predictions = pd.DataFrame(data.iloc[:, 0]), pd.DataFrame(data.iloc[:, 1])
    return auc_analysis.calculate_auc_scikit(labels, predictions)

def load_data(label_path, prediction_path):
    print(f"Lade Daten aus: {label_path}")
    labels = data_loader.load_data(label_path)

    print(f"Lade Daten aus: {prediction_path}")
    predictions = data_loader.load_data(prediction_path)

    data = data_loader.merge_df(labels, predictions)

    return data

def secure_auc(data, n_steps=100, epsilon=0):
    thresholds = np.linspace(1, 0, n_steps)
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
    repetitions = int(-(np.log(1/len(data[0])/2))/np.log(2))
    return mpc.calculate_AUC(TP, FP, P, N, epsilon, repetitions)

def dp_auc_calc(data, n_steps=1000, epsilon=0):
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

    TP = []
    FP = []
    P = []
    N = []

    thresholds = np.linspace(1, 0, n_steps)

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

    return mpc.compute_AUC_without_mpc(TP, FP, P, N, epsilon)

def to_scalar(x):
    x = np.asarray(x)
    if x.size == 1:
        return float(x.item())
    return float(x.flatten()[0])


def vary_thresholds():
    mpc_auc = []
    dp_auc = []
    real_auc = []

    epsilons = [0]

    y_max = [0.02, 0.03, 0.015, 0.02]


    for i, (labels, predictions) in enumerate(paths):
        data = load_data(labels, predictions)
        real_auc_tmp = calc_scikit_auc(data)
        real_auc.append(real_auc_tmp)

        data = data_loader.split_shuffled_df(data, 2)

        secure_mpc_auc = []
        for t in steps:
            secure_mpc_auc_step = []
            for e in epsilons:
                secure_auc_tmp = secure_auc(data, t, e)[0]
                result = np.abs(secure_auc_tmp - real_auc_tmp)
                secure_mpc_auc_step.append(result)
            secure_mpc_auc.append(secure_mpc_auc_step)
        secure_mpc_auc = np.array(secure_mpc_auc)
        mpc_auc.append(secure_mpc_auc)

        secure_dp_auc = []
        for t in steps:
            secure_dp_auc_step = []
            for e in epsilons:
                secure_dp_auc_tmp = dp_auc_calc(data, t, e)
                result = np.abs(secure_dp_auc_tmp - real_auc_tmp)
                secure_dp_auc_step.append(result)
            secure_dp_auc.append(secure_dp_auc_step)
        secure_dp_auc = np.array(secure_dp_auc)
        dp_auc.append(secure_dp_auc)

        # Plot one line per epsilon
        plt.figure()

        for j, e in enumerate(epsilons):
            plt.plot(
                steps,
                secure_mpc_auc[:, j],
                marker="o",
                label=f"MPC AUC"
            )
            plt.plot(
                steps,
                secure_dp_auc[:, j],
                marker="x",
                linestyle="--",
                label=f"Noiseless AUC"
            )

        plt.xlabel("threshold steps")
        plt.ylabel("absolute error")
        plt.title(f"AUC error vs threshold steps ({dataset[i]})")

        if i < len(y_max):
            plt.ylim(0, y_max[i])

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../plots/auc_thresholds_{dataset[i]}.png", dpi=300)
        plt.close()


def vary_epsilon(noise_samples=60):
    for i, (labels, predictions) in enumerate(paths):
        data = load_data(labels, predictions)
        real_auc = float(calc_scikit_auc(data))
        data = data_loader.split_shuffled_df(data, 2)

        mpc_auc = []
        dp_auc = []

        for e in epsilons:
            mpc_eps = []
            dp_eps = []

            for _ in range(noise_samples):
                mpc_val = to_scalar(secure_auc(data, 100, e))
                dp_val = to_scalar(dp_auc_calc(data, 100, e))

                mpc_eps.append(abs(mpc_val - real_auc))
                dp_eps.append(abs(dp_val - real_auc))

            mpc_auc.append(mpc_eps)
            dp_auc.append(dp_eps)

        # each entry in these lists is one box
        mpc_data = [np.asarray(x, dtype=float) for x in mpc_auc]
        dp_data = [np.asarray(x, dtype=float) for x in dp_auc]

        plt.figure(figsize=(10, 6))

        positions_mpc = np.arange(len(epsilons)) * 2.0
        positions_dp = positions_mpc + 0.7

        bp1 = plt.boxplot(
            mpc_data,
            positions=positions_mpc,
            widths=0.5,
            patch_artist=True
        )

        bp2 = plt.boxplot(
            dp_data,
            positions=positions_dp,
            widths=0.5,
            patch_artist=True
        )

        # color the boxes
        for box in bp1['boxes']:
            box.set_facecolor("steelblue")

        for box in bp2['boxes']:
            box.set_facecolor("indianred")

        plt.xticks(
            positions_mpc + 0.35,
            [str(e) for e in epsilons]
        )

        plt.xlabel("epsilon")
        plt.ylabel("absolute error")
        plt.title(f"AUC error distribution vs epsilon ({dataset[i]})")

        legend_handles = [
            Patch(facecolor="steelblue", label="Secure MPC"),
            Patch(facecolor="indianred", label="DP")
        ]

        plt.legend(handles=legend_handles)

        plt.tight_layout()
        plt.savefig(f"../plots/auc_epsilon_boxplot_{dataset[i]}.png", dpi=300)
        plt.close()

def vary_parties():
    threshold_steps = 100
    party_counts = [2, 3, 4, 6, 8]
    selected_epsilons = [0, 0.3, 1.0, 3, 9]

    noise_repeats = 20

    for i, (labels, predictions) in enumerate(paths):
        data = load_data(labels, predictions)
        real_auc = float(calc_scikit_auc(data))

        # store results by epsilon
        mpc_means = []
        mpc_stds = []
        dp_means = []
        dp_stds = []

        for e in selected_epsilons:
            mpc_eps_means = []
            mpc_eps_stds = []
            dp_eps_means = []
            dp_eps_stds = []

            for p in party_counts:
                mpc_all_errors = []
                dp_all_errors = []


                split_data = data_loader.split_shuffled_df(data, p)

                for _ in range(noise_repeats):
                    mpc_val = float(to_scalar(secure_auc(split_data, threshold_steps, e)))
                    dp_val = float(to_scalar(dp_auc_calc(split_data, threshold_steps, e)))

                    mpc_all_errors.append(abs(mpc_val - real_auc))
                    dp_all_errors.append(abs(dp_val - real_auc))

                mpc_eps_means.append(np.mean(mpc_all_errors))
                mpc_eps_stds.append(np.std(mpc_all_errors))
                dp_eps_means.append(np.mean(dp_all_errors))
                dp_eps_stds.append(np.std(dp_all_errors))

            mpc_means.append(mpc_eps_means)
            mpc_stds.append(mpc_eps_stds)
            dp_means.append(dp_eps_means)
            dp_stds.append(dp_eps_stds)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        for j, e in enumerate(selected_epsilons):
            axes[0].errorbar(
                party_counts,
                mpc_means[j],
                yerr=mpc_stds[j],
                marker='o',
                capsize=3,
                label=f"ε={e}"
            )
            axes[1].errorbar(
                party_counts,
                dp_means[j],
                yerr=dp_stds[j],
                marker='o',
                capsize=3,
                label=f"ε={e}"
            )

        axes[0].set_title("MPC")
        axes[1].set_title("DP")

        for ax in axes:
            ax.set_xlabel("number of parties")
            ax.set_xticks(party_counts)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("mean absolute error")
        fig.suptitle(f"AUC error vs number of parties ({dataset[i]})")
        axes[1].legend(title="privacy budget")

        plt.tight_layout()
        plt.savefig(f"../plots/auc_parties_{dataset[i]}.png", dpi=300)
        plt.close()

if __name__ == '__main__':
    args = parse_args()

    vary_thresholds()
    #vary_epsilon()
    #vary_parties()