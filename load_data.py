import numpy as np
import time
import scipy.io

def load(file_path_E,file_path_G):
    data = scipy.io.loadmat(file_path_E)
    E = data['E'].transpose()
    E *= 2

    data = scipy.io.loadmat(file_path_G)
    G = data['G'].transpose()
    return G,E,E.shape[1]