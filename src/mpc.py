import crypten
import pandas as pd
import torch
from crypten import cryptensor


# compare = x[i] <= y[i]
#
# classification = 1 * compare + 0 * (1-compare)

@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment_approx(labels, predictions, threshold):
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

    def compute_AUC(fpr, tpr):
        sum = crypten.cryptensor(torch.tensor([0]))
        for i in range(1, len(tpr)):
            part1 = tpr[i]+tpr[i-1]
            part2 = fpr[i]-fpr[i-1]
            #print(part1.get_plain_text())
            #print(part2.get_plain_text())
            sum += (part1 * part2) / 2
        crypten.print("sum", sum.get_plain_text())

    def newton_raphson(x, a, b, num=2):

        for i in range(num):
            temp = x * b
            temp = 2 - temp
            x = x * temp

        return x * a

    labels_enc = encrypt(labels)
    predictions_enc = encrypt(predictions)

    fpr, tpr = [], []

    for t in threshold:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        values = torch.tensor([TP, TN, FP, FN])
        sec_values = crypten.cryptensor(values)

        classified = SEC_classifier(predictions_enc, t)
        for i in range(len(predictions)):
            sec_values[0] += labels_enc[i] * classified[i] #TP
            sec_values[1] += (1 - labels_enc[i]) * (1 - classified[i]) #TN
            sec_values[2] += (1 - labels_enc[i]) * classified[i] #FP
            sec_values[3] += labels_enc[i] * (1 - classified[i]) #FN

        # calculate TPR & FPR
        crypten.print("values", sec_values.get_plain_text())
        #TP, TN, FP, FN = sec_values.get_plain_text()
        #TPR = sec_values[0] / (sec_values[0] + sec_values[3])
        #FPR = sec_values[2] / (sec_values[2] + sec_values[1])
        TPR = newton_raphson(0.002, sec_values[0], (sec_values[0] + sec_values[3]) )
        FPR = newton_raphson(0.002, sec_values[2], (sec_values[2] + sec_values[1]) )


        fpr.append(FPR)
        tpr.append(TPR)
        print(f"TPR: {TPR.get_plain_text()}, FPR: {FPR.get_plain_text()}")

    compute_AUC(fpr, tpr)



@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment(data):

    def sortMergeJoin(left_id, left_pred, right_id, right_pred):
        result = []

        left_index = 0
        right_index = 0

        while left_index < len(left_pred) and right_index < len(right_pred):

            compare = left_pred[left_index] >= right_pred[right_index]
            val = compare * left_pred[left_index] + (1-compare) * right_pred[right_index]
            result.append(val.get_plain_text())

            compare = compare.get_plain_text()
            left_index += compare
            right_index += (1 - compare)

        crypten.print(result)


    left = data[0]
    crypten.print(left)
    right = data[1]
    crypten.print(right)

    left_id = left.index.tolist()
    right_id = right.index.tolist()
    left_pred = torch.tensor(left.iloc[:, 1].tolist())
    right_pred = torch.tensor(right.iloc[:, 1].tolist())

    left_id = crypten.cryptensor(torch.tensor(left_id))
    right_id = crypten.cryptensor(torch.tensor(right_id))
    left_pred = crypten.cryptensor(left_pred)
    right_pred = crypten.cryptensor(right_pred)

    sortMergeJoin(left_id, left_pred, right_id, right_pred)

    # def SEC_classifier(prediction, t):
    #     t = torch.tensor(t)
    #     t_enc = cryptensor(t)
    #
    #     # iterate through prediction and classify as
    #     # 1 or 0 according to given threshold
    #     # for x in prediction:
    #     #    compare = x >= t_enc
    #     #    classification = 1 * compare + 0 * (1-compare)
    #     #    res.append(classification)
    #
    #     compare = prediction >= t_enc
    #
    #     return compare
    #
    # def compute_AUC(fpr, tpr):
    #     sum = crypten.cryptensor(torch.tensor([0]))
    #     for i in range(1, len(tpr)):
    #         part1 = tpr[i]+tpr[i-1]
    #         part2 = fpr[i]-fpr[i-1]
    #         #print(part1.get_plain_text())
    #         #print(part2.get_plain_text())
    #         sum += (part1 * part2) / 2
    #     crypten.print("sum", sum.get_plain_text())
    #
    # def newton_raphson(x, a, b, num=2):
    #
    #     for i in range(num):
    #         temp = x * b
    #         temp = 2 - temp
    #         x = x * temp
    #
    #     return x * a
    #
    # labels_enc = encrypt(labels)
    # predictions_enc = encrypt(predictions)
    #
    # fpr, tpr = [], []
    #
    # for t in threshold:
    #     TP = 0
    #     TN = 0
    #     FP = 0
    #     FN = 0
    #     values = torch.tensor([TP, TN, FP, FN])
    #     sec_values = crypten.cryptensor(values)
    #
    #     classified = SEC_classifier(predictions_enc, t)
    #     for i in range(len(predictions)):
    #         sec_values[0] += labels_enc[i] * classified[i] #TP
    #         sec_values[1] += (1 - labels_enc[i]) * (1 - classified[i]) #TN
    #         sec_values[2] += (1 - labels_enc[i]) * classified[i] #FP
    #         sec_values[3] += labels_enc[i] * (1 - classified[i]) #FN
    #
    #     # calculate TPR & FPR
    #     crypten.print("values", sec_values.get_plain_text())
    #     #TP, TN, FP, FN = sec_values.get_plain_text()
    #     #TPR = sec_values[0] / (sec_values[0] + sec_values[3])
    #     #FPR = sec_values[2] / (sec_values[2] + sec_values[1])
    #     TPR = newton_raphson(0.002, sec_values[0], (sec_values[0] + sec_values[3]) )
    #     FPR = newton_raphson(0.002, sec_values[2], (sec_values[2] + sec_values[1]) )
    #
    #
    #     fpr.append(FPR)
    #     tpr.append(TPR)
    #     print(f"TPR: {TPR.get_plain_text()}, FPR: {FPR.get_plain_text()}")
    #
    # compute_AUC(fpr, tpr)



