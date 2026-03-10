import torch
import crypten

from src import data_loader
from src.data_loader import split_df, merge_df

predictions = data_loader.load_data("../data/pred_cons_100.txt")
labels = data_loader.load_data("../data/labels_100.txt")
df = merge_df(labels, predictions)
predictions = split_df(df, 2)

print(type(predictions[0].iloc[0]))


#@crypten.mpc.run_multiprocess(world_size=2)
#def run_experiment(predictions):
