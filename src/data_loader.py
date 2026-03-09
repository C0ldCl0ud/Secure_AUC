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

