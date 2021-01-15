from matplotlib import pyplot
import numpy as np

def decision_tree_fit(x, y, n_classes = None, max_depth = None, min_samples_split = 2):
    gini = 1
    features = list()
    threshold = list()
    depth = 0
    gini, left, right, features, threshold = best_split(x, y)
    nodes = {"gini": gini, "samples": len(x), "value": n_classes, "features": features, 
                "threshold": threshold, "left": left[0], "threshold": right[0]}
    return nodes

def best_split(x, y):
    gini = 1
    best_gini = np.inf
    features = []
    threshold = []
    m = len(x[0])
    n = len(x)
    for f in range(m):
        feature = np.unique(x[:, f])
        for t in range(len(feature)):
            theta = feature[t]
            x_l = list()
            x_r = list()
            y_l = list()
            y_r = list()
            for i in range(n):
                if x[i][f] <= theta:
                    x_l.append(x[i])
                    y_l.append(y[i])
                    features.append(feature)
                    threshold.append(theta)
                else:
                    x_r.append(x[i])
                    y_r.append(y[i])
                    features.append(feature)
                    threshold.append(theta)
            gini = impurity(x_l, x_r)
            if gini < best_gini:
                best_gini = gini
                best_f = i
                best_theta = t
                best_left = (x_l, y_l)
                best_right = (x_r, y_r)
    return gini, best_left, best_right, features, threshold

def impurity(left, right):
    t_l = [l[1] for l in left]
    all_l = list(np.unique(t_l))

    t_r = [r[1] for r in right]
    all_r = list(np.unique(t_r))
    
    g_i = ((len(left) * gini_index(left, all_l)) + (len(right) * gini_index(right, all_r))) / len(left) + len(right)
    return g_i

def gini_index(t, all_t):
    sum = 0
    for a in all_t:
        sum += (a / len(t)) ** 2
    return 1 - sum

def decision_tree_predict(dt, x):
    maxes = []
    for i in x:
        maxes.append(predict(dt, i))
    return maxes

def predict(dt, x):
    if ((dt["left"] == None) and (dt["right"] == None)):
        return np.argmax(dt["value"])
    elif x[dt["feature"]] <= dt["threshold"]:
        return predict(dt["left"], x)
    else:
        return predict(dt["right"], x)
