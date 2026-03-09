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
import auc_analysis
import data_loader
import mpc
import utils
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import crypten
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = ","
#device = torch.device("cpu")
#print(f"Using device: {device}")

paths = {
    "../data/labels_100.txt": "../data/pred_cons_100.txt",
    "../data/labels_1000.txt": "../data/pred_cons_1000.txt",
    "../data/labels_10000.txt": "../data/pred_cons_10000.txt",
    "../data/labels_100000.txt": "../data/pred_cons_100000.txt",
}

def main():
    def auc(labels, predictions):

        print(f"Lade Daten aus: {labels}")
        labels = data_loader.load_data(labels)

        print(f"Lade Daten aus: {predictions}")
        predictions = data_loader.load_data(predictions)
        labels_enc = mpc.encrypt(labels)
        predictions_enc = mpc.encrypt(predictions)

        auc_scikit = auc_analysis.calculate_auc_scikit(labels, predictions)

        threshold = [0, 0.2, 0.5, 1]

        for t in threshold:
            mpc.SEC_rocCalc(predictions_enc, labels_enc, t)


    for labels, predictions in paths.items():
        auc(labels, predictions)








if __name__ == '__main__':
    main()
