from .A import *
import numpy as np

def calculate_y(X,W,V,U, B):
    W_prime = np.array(W)
    U_prime = np.array(U)
    V_prime = np.array(V)
    W_prime.reshape([2,1])
    V_prime.reshape([2,1])
    Z = np.array([sigmoid(np.matmul(X,W_prime) + B[0]), sigmoid(np.matmul(X, U_prime) + B[1])])
    U_prime.reshape([2,1])
    return sigmoid(np.matmul(Z, U_prime) + B[2])

W = [np.random.rand(), np.random.rand()]
V = [np.random.rand(), np.random.rand()]
U = [np.random.rand(), np.random.rand()]

for i in range(n_epoch):
    grad_w = np.zeros([len(W)])
    grad_v = np.zeros([len(V)])
    grad_u = np.zeros([len(U)])