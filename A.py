import numpy as np
from read_data import get_file_array
import matplotlib.pyplot as plt
import matplotlib

def sigmoid(x):
    return 1/(1 + np.exp(-x))




def dy_dall(x, w, b):
    wx_plus_b = w*x + b
    return x*sigmoid(wx_plus_b)*(1 - sigmoid(wx_plus_b))

def calculate_y(W,X,b):
    w2 = np.array(W)
    x2 = np.array(X)
    w2.reshape([2,1])
    np.matmul(x2, w2) + b
    return np.matmul(X,W) + b

def calculate_cost(y, y_prim):
    return 1/2*((y - y_prim)**2)
# define input symbols
X, y_array = get_file_array('./data/data.csv')

# define variable symbols
W = [np.random.rand(),np.random.rand()]
b = np.random.rand()

lr = 0.01
train = 150
test = 50


def A_training(n_epoch):
    for i in range(n_epoch):
        grad = np.zeros([len(W)])
        for r in range(len(W)):
            for j in range(train):
                y = calculate_y(W,X[j],b)
                cost = calculate_cost(y_array[j], y)
                dcost_dw = (y - y_array[j]) * dy_dall(W[r], X[j][r], b)
                grad[r] += dcost_dw
        for r in range(len(W)):
            W[r] = W[r] - lr*grad[r]
    lost = 0
    for i in range(train, train + test):
        y = calculate_y(W,X[i],b)
        lost += 1 if np.abs(y - y_array[i]) > 0.2 else 0
    return lost
epoch_array = [i*3 for i in range(100)]
result = [A_training(item) for item in epoch_array]
fig, ax = plt.subplots()
ax.plot(epoch_array, result)
ax.set(xlabel='epoch size', ylabel='lost', title='NN A result with different lr')
ax.grid()
fig.savefig('A_epoch(3-300)')
plt.show()
print(np.mean(result))