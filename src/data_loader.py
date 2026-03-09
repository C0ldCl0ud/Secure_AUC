import pandas as pd

def load_data(path):
    df = pd.read_csv(path, dtype=float, header=None)
    return df

