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
from pandas.io.xml import preprocess_data

import auc_analysis
import data_loader
import mpc
import utils
import multiprocessing

multiprocessing.set_start_method("fork", force=True)
import time
import numpy as np

import crypten
import torch
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
    "../data/labels_100.txt": "../data/pred_cons_100.txt"
}

def approx_auc(paths, n_steps=100):

    def auc(labels, predictions):


        print(f"Lade Daten aus: {labels}")
        labels = data_loader.load_data(labels)

        print(f"Lade Daten aus: {predictions}")
        predictions = data_loader.load_data(predictions)

        thresholds = np.linspace(1, 0, n_steps)

        mpc.run_experiment_approx(labels, predictions, thresholds)
        auc_scikit = auc_analysis.calculate_auc_scikit(labels, predictions)

    for labels, predictions in paths.items():
        auc(labels, predictions)



def real_auc(paths):

    def auc(labels, predictions):

        print(f"Lade Daten aus: {labels}")
        labels = data_loader.load_data(labels)

        print(f"Lade Daten aus: {predictions}")
        predictions = data_loader.load_data(predictions)

        data = data_loader.merge_df(labels, predictions)
        data = data_loader.split_df(data, 2)

        mpc.run_experiment(data)
        auc_scikit = auc_analysis.calculate_auc_scikit(labels, predictions)

    for labels, predictions in paths.items():
        auc(labels, predictions)

if __name__ == '__main__':
    #print("-------------------------------------------------------------------------")
    #approx_auc(paths_full)
    #print("-------------------------------------------------------------------------")
    #real_auc()



