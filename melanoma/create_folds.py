import os

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    input_path = 'data/'
    df = pd.read_csv(os.path.join(input_path,'train.csv'))
    print(df.head())

    




