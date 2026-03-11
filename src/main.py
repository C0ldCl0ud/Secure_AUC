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
import mpc
import numpy as np

import multiprocessing
multiprocessing.set_start_method("fork", force=True)

#os.environ["CUDA_VISIBLE_DEVICES"] = ","
#device = torch.device("cpu")
#print(f"Using device: {device}")

paths_full = {
    "../data/labels_100.txt": "../data/pred_cons_100.txt",
    "../data/labels_1000.txt": "../data/pred_cons_1000.txt",
    "../data/labels_10000.txt": "../data/pred_cons_10000.txt",
    "../data/labels_100000.txt": "../data/pred_cons_100000.txt",
}
paths_demo = {
    "../data/labels_10000.txt": "../data/pred_cons_10000.txt"
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

def secure_auc(data, approx=False, n_steps=1000, partitions=1):
    if approx:
        data = data.sample(frac=1, random_state=42)
        thresholds = np.linspace(1, 0, n_steps)
        mpc.run_experiment_approx(data, thresholds, partitions)
    else:
        data = data_loader.split_shuffled_df(data, 2)
        mpc.run_experiment_real(data)


if __name__ == '__main__':
    for labels_path, predictions_path in paths_demo.items():

        data = load_data(labels_path, predictions_path)

        print("-------------------------------------------------------------------------")
        print("Calculating accurate AUC:")
        secure_auc(data)
        print("-------------------------------------------------------------------------")
        print("Calculating approximate AUC:")
        secure_auc(data, approx=True, n_steps=100, partitions=10)
        print("-------------------------------------------------------------------------")
        print("Calculating scikit AUC:")
        calc_scikit_auc(data)
