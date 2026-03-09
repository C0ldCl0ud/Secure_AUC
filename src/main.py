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
import utils
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import crypten
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = ","
#device = torch.device("cpu")
#print(f"Using device: {device}")

def main():
    pass

if __name__ == '__main__':
    main()
