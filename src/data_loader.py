import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path, dtype=float, header=None)
    return df

def merge_df(df1, df2):
    return pd.concat([df1, df2], axis=1)

def split_df(df, number_of_splits):
    dataframes = []
    for i in range(number_of_splits):
        dataframe = df.iloc[i::number_of_splits]
        dataframes.append(dataframe)
    return dataframes

def split_shuffled_df(df, number_of_splits):
    shuffeled = df.sample(frac=1)

    split = split_df(shuffeled, number_of_splits)
    for i in range(number_of_splits):
        split[i] = split[i].iloc[split[i].iloc[:, 1].argsort()[::-1]]

    return split

