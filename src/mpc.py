import crypten
import torch

from src.auc_analysis import statistics, classifier

@crypten.mpc.run_multiprocess(world_size=2)
def encrypt(vector):
    # transform
    values = vector.iloc[:, 0].tolist()
    x = torch.tensor(values)
    x_enc = crypten.cryptensor(x)
    return x_enc

@crypten.mpc.run_multiprocess(world_size=2)
def SEC_calcROC(prediction, truth, t):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    values = torch.tensor([TP, TN, FP, FN])
    sec_values = crypten.cryptensor(values)

    # apply classifier
    classified = SEC_classifier(prediction, t)

    if len(prediction) != len(truth):
        raise Exception("Prediction and Reference have unequal length.")

    # compare classified to truth/ref and calc TP/TN/FP/FN
    #print(classified)
    #print(truth.iloc[:, 0])
    for i in range(len(prediction)):
        sec_values[0] += truth[i] * classified[i]
        sec_values[1] += (1 - truth[i]) * (1 - classified[i])
        sec_values[2] += (1 - truth[i]) * classified[i]
        sec_values[3] += truth[i] * (1 - classified[i])

    # calculate TPR & FPR
    crypten.print("values", sec_values.get_plain_text())
    TPR = sec_values[0] / (sec_values[0] + sec_values[3])
    FPR = sec_values[2] / (sec_values[2] + sec_values[1])

    crypten.print("TPR", TPR.get_plain_text())
    crypten.print("FPR", FPR.get_plain_text())
    # put together as ROC-DF
    #result.ROC_df.assign(FPR=result.FPR, TPR=result.TPR)

    return 0


def SEC_classifier(prediction, t):
    res = []

    # iterate through prediction and classify as
    # 1 or 0 according to given threshold
    for x in prediction:
        compare = x >= t
        classification = 1 * compare + 0 * (1-compare)
        res.append(classification)

    return res

# compare = x[i] <= y[i]
#
# classification = 1 * compare + 0 * (1-compare)
