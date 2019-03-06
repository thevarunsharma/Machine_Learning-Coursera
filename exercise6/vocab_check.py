import numpy as np
import pandas as pd

vocab=pd.read_csv("vocab.txt",header=None,sep=None,engine='python')

def get_index(s):
        df=vocab[vocab[1]==s].as_matrix()
        if len(df)!=0:
            return df[0][0]
        else:
            return None
