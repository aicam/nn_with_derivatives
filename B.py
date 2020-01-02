from read_data import get_file_array
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dy_dall(x, w, b):
    wx_plus_b = w * x + b
    return x * sigmoid(wx_plus_b) * (1 - sigmoid(wx_plus_b))


def calculate_y(W, X, b):
    w2 = np.array(W)
    x2 = np.array(X)
    w2.reshape([2, 1])
    np.matmul(x2, w2) + b
    return np.matmul(X, W) + b


def calculate_cost(y, y_prim):
    return 1 / 2 * ((y - y_prim) ** 2)


# define input symbols
X, y_array = get_file_array('./data/data.csv')

# define variable symbols
W = [np.random.rand(), np.random.rand()]
b = np.random.rand()

n_epoch = 50
lr = 0.01
train = 150
test = 50


def calculate_y0(X, W, V, U, B):
    W_prime = np.array(W)
    U_prime = np.array(U)
    V_prime = np.array(V)
    W_prime.reshape([2, 1])
    V_prime.reshape([2, 1])
    Z = np.array([sigmoid(np.matmul(X, W_prime) + B[0]), sigmoid(np.matmul(X, U_prime) + B[1])])
    U_prime.reshape([2, 1])
    return sigmoid(np.matmul(Z, U_prime) + B[2])


W = [np.random.rand(), np.random.rand()]
V = [np.random.rand(), np.random.rand()]
U = [np.random.rand(), np.random.rand()]
B = [np.random.rand(), np.random.rand(), np.random.rand()]
def B_training():
    for i in range(n_epoch):
        grad_w = np.zeros([len(W)])
        grad_v = np.zeros([len(V)])
        grad_u = np.zeros([len(U)])
        for r1 in range(len(U)):
            for r2 in range(len(W)):
                for r3 in range(len(V)):
                    for j in range(train):
                        y0 = calculate_y0(X[j], W, V, U, B)
                        cost = calculate_cost(y_array[j], y0)
                        # calculate_y is also used to calculate WX+b0
                        dcost_du = (y0 - y_array[j]) * dy_dall(calculate_y(W, X[j], B[0]), U[r1], B[0])

                        grad_u[r1] += dcost_du/3
                        dcost_dw = (y0 - y_array[j]) * dy_dall(calculate_y(W, X[j], B[0]), U[r1], B[0]) * dy_dall(X[j][r2], W[r2],
                                                                                                            B[0])
                        grad_w[r2] += dcost_dw/3
                        dcost_dv = (y0 - y_array[j]) * dy_dall(calculate_y(V, X[j], B[0]), U[r1], B[0]) * dy_dall(X[j][r3], V[r3],
                                                                                                            B[1])
                        grad_v[r3] += dcost_dv/3
        for r1 in range(len(U)):
            U[r1] = U[r1] - lr * grad_u[r1]
        for r1 in range(len(V)):
            V[r1] = V[r1] - lr * grad_v[r1]
        for r1 in range(len(W)):
            W[r1] = W[r1] - lr * grad_w[r1]
    lost = 0
    for i in range(train, train + test):
        y = calculate_y0(X[i], W, V, U, B)
        lost += not np.logical_xor(y, y_array[i])
    return lost
