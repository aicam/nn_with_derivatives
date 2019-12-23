import random
def get_file_array(path):
    f = open(path)
    X = []
    y_arr = []
    for l in f:



        X.append([float(l.split(',')[0]), float(l.split(',')[1])])
        y_arr.append(int(l.split(',')[2]))
    z = list(zip(X,y_arr))
    random.shuffle(z)
    X, y_arr = zip(*z)
    return X, y_arr