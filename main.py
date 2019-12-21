from sympy import *
import numpy as np
import math
from read_data import get_file_array
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dy_dw(x,w,b):
    wx_plus_b = np.sum(np.array(x)*np.array(w)) + b
    return sigmoid(wx_plus_b)*(1 - sigmoid(wx_plus_b))

# define input symbols
X = []

# define variable symbols
W = [np.random.rand(),np.random.rand()]
B = [np.random.rand()]

# define output symbols
Y = []

COST = math.inf
x0_array, x1_array, y_array = get_file_array('./data/data.csv')




