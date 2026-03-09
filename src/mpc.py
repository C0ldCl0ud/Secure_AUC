import crypten
import torch
from crypten import cryptensor

from src.auc_analysis import statistics, classifier





# compare = x[i] <= y[i]
#
# classification = 1 * compare + 0 * (1-compare)

@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment(labels, predictions, threshold):
    if len(predictions) != len(labels):
        raise Exception("Prediction and Reference have unequal length.")

    def encrypt(vector):
        # transform
        values = vector.iloc[:, 0].tolist()
        x = torch.tensor(values)
        x_enc = crypten.cryptensor(x)
        return x_enc

    def SEC_classifier(prediction, t):
        t = torch.tensor(t)
        t_enc = cryptensor(t)

        # iterate through prediction and classify as
        # 1 or 0 according to given threshold
        # for x in prediction:
        #    compare = x >= t_enc
        #    classification = 1 * compare + 0 * (1-compare)
        #    res.append(classification)

        compare = prediction >= t_enc

        return compare

    labels_enc = encrypt(labels)
    predictions_enc = encrypt(predictions)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    values = torch.tensor([TP, TN, FP, FN])
    sec_values = crypten.cryptensor(values)

    classified = SEC_classifier(predictions_enc, threshold)

    for i in range(len(predictions)):
        sec_values[0] += labels_enc[i] * classified[i]
        sec_values[1] += (1 - labels_enc[i]) * (1 - classified[i])
        sec_values[2] += (1 - labels_enc[i]) * classified[i]
        sec_values[3] += labels_enc[i] * (1 - classified[i])

    # calculate TPR & FPR
    crypten.print("values", sec_values.get_plain_text())
    TPR = sec_values[0] / (sec_values[0] + sec_values[3])
    FPR = sec_values[2] / (sec_values[2] + sec_values[1])

    crypten.print("TPR", TPR.get_plain_text())
    crypten.print("FPR", FPR.get_plain_text())


