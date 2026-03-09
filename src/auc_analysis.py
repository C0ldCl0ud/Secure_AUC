from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import data_loader
import utils


paths = {
    "../data/labels_100.txt": "../data/pred_cons_100.txt",
    "../data/labels_1000.txt": "../data/pred_cons_1000.txt",
    "../data/labels_10000.txt": "../data/pred_cons_10000.txt",
    "../data/labels_100000.txt": "../data/pred_cons_100000.txt",
}

@dataclass
class statistics:
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    threshold = 0
    TPR = []
    FPR = []
    ROC_df = pd.DataFrame()
    reference = 0

def calcROC(prediction, truth, t):
    result = statistics()
    result.threshold = t

    # apply classifier
    classified = classifier(prediction, t)

    if len(prediction) != len(truth):
        raise Exception("Prediction and Reference have unequal length.")

    # compare classified to truth/ref and calc TP/TN/FP/FN
    print(classified)
    print(truth.iloc[:,0])
    for i in range(len(prediction)):
        if classified[i] == truth.iloc[i,0]:
           if truth.iloc[i,0] == 1:
               result.TP += 1
           else:
               result.TN += 1
        else:
            if classified[i] == 1:
                result.FP += 1
            else:
                result.FN += 1

    # calculate TPR & FPR
    print("TP: ", result.TP)
    print("FP: ", result.FP)
    print("FN: ", result.FN)
    print("TN: ", result.TN)
    result.TPR = result.TP / (result.TP + result.FN)
    result.FPR = result.FP / (result.TN + result.FP)

    # put together as ROC-DF
    result.ROC_df.assign(FPR = result.FPR, TPR = result.TPR)

    # add reference
    result.reference = roc_curve(truth, prediction)

    return result


def classifier(prediction, t):
    res = []

    # iterate through prediction and classify as
    # 1 or 0 according to given threshold
    for x in prediction.iloc[:, 0]:
        if x >= t:
            res.append(1)
        else:
            res.append(0)

    return res
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

def calculate_auc_scikit(truth, prediction):
    auc = roc_auc_score(truth, prediction)
    print("AUC: ", auc)

def calculate_scilearn_all_auc():

    for labels, prediction in paths.items():
        labels = data_loader.load_data(labels)
        prediction = data_loader.load_data(prediction)

        calculate_auc_scikit(labels, prediction)

        fpr, tpr, threshold = roc_curve(labels, prediction)
        utils.plot_roc(fpr, tpr)

        fpr, tpr = [], []
        for t in thresholds:
            statistics = calcROC(prediction, labels, t)
            fpr.append(statistics.FPR)
            tpr.append(statistics.TPR)
        utils.plot_roc(fpr, tpr)

calculate_scilearn_all_auc()


