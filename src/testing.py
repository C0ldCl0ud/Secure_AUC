import torch
import crypten
import matplotlib.pyplot as plt

from src import data_loader
from src.data_loader import split_df, merge_df
from src.utils import secure_sort

predictions = data_loader.load_data("../data/pred_cons_100.txt")
labels = data_loader.load_data("../data/labels_100.txt")
df = merge_df(labels, predictions)
predictions = split_df(df, 2)



@crypten.mpc.run_multiprocess(world_size=2)
def run_experiment(predictions):
    #crypten.print(type(predictions[0].iloc[0].tolist()))
    tmp = []

    #crypten.print(len(predictions))
    i = 0
    while i < len(predictions):
        #crypten.print(i)
        for j in range(len(predictions[i])):
            tmp.append(predictions[i].iloc[j].tolist())
        i += 1

    #crypten.print(tmp)
    encryption = crypten.cryptensor(tmp)
    #crypten.print(encryption[0])

    def secure_sort(x_enc):
        result = x_enc
        n = len(result)

        for i in range(n - 1):
            for j in range(0, n - i - 1):
                compare = result[j] >= result[j + 1]  # 1 if left <= right
                crypten.print("J:", result[j].get_plain_text())
                crypten.print("J+1:", result[j+1].get_plain_text())
                crypten.print(compare.get_plain_text())

                left = result[j]
                right = result[j + 1]

                new_left = left * compare[1] + right * (1 - compare[1])
                crypten.print("new left:", new_left.get_plain_text())
                new_right = right * compare[1] + left * (1 - compare[1])
                crypten.print("new right:", new_right.get_plain_text())

                result[j] = new_left
                result[j + 1] = new_right

        crypten.print(result.get_plain_text())
        return result

    secure_sort(encryption)

run_experiment(predictions)