from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import roc_curve


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
    for i in range(len(prediction)):
        if classified[i] == truth[i]:
           if truth[i] == 1:
               result.TP += 1
           else:
               result.TN += 1
        else:
            if classified[i] == 1:
                result.FP += 1
            else:
                result.FN += 1

    # calculate TPR & FPR
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
    for x in prediction:
        if float(x) >= t:
            res.append(1)
        else:
            res.append(0)

    return res