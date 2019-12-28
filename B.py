from .A import *
import numpy as np

def calculate_y0(X,W,V,U, B):
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
B = [np.random.rand(), np.random.rand(), np.random.rand()]
for i in range(n_epoch):
    grad_w = np.zeros([len(W)])
    grad_v = np.zeros([len(V)])
    grad_u = np.zeros([len(U)])
    for r1 in range(len(U)):
        for r2 in range(len(W)):
            for r3 in range(len(V)):
                for j in range(train):
                    y0 = calculate_y0(X[j],W,V,U,B)
                    cost = calculate_cost(y_array[j],y0)
                    # calculate_y is also used to calculate WX+b0
                    dcost_du = (y0 - y_array[j])*dy_dall(calculate_y(W,X,B[0]),U[r1],B[0])
                    grad_u[r1] += dcost_du
                    dcost_dw = (y0 - y_array)*dy_dall(calculate_y(W,X,B[0]),U[r1],B[0])*dy_dall(X[j][r2],W[r2],B[0])
                    grad_w[r2] += dcost_dw
                    dcost_dv = (y0 - y_array)*dy_dall(calculate_y(V,X,B[0]),U[r1],B[0])*dy_dall(X[j][r3],V[r3],B[1])
                    grad_v[r3] += dcost_dv
    

