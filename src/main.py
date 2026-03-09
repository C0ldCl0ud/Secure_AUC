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
import data_loader
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
    for labels, predictions in paths.items():
        labels = data_loader.load_data(paths)
        predictions = data_loader.load_data(paths)



if __name__ == '__main__':
    main()
