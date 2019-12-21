def get_file_array(path):
    f = open(path)
    x0_arr = []
    x1_arr = []
    y_arr = []
    for l in f:



        x0_arr.append(l.split(',')[0])
        x1_arr.append(l.split(',')[1])
        y_arr.append(l.split(',')[2])
    return x0_arr, x1_arr, y_arr