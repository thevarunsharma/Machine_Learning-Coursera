import numpy as np
import pandas as pd

def gaussian_kernel(x1,x2):
    sigma=2
    return np.exp(-(np.sum((x1-x2)**2)/(2*sigma**2)))
