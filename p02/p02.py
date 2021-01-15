import math

def net_input(x, w):
    vec = []
    for u in x:
        prod = 0
        for ui, vi in zip(u,w):
            prod += ui * vi
        vec.append(prod)
    return vec

""" Test """
#a = [[1,2],[3,4],[-1,2.5]]
#b = [1,-1]
#c = [2, 0.5]
#print(net_input(a, c))
# Should return [-1.0, -1.0, -3.5]
# Should return [3.0, 8.0, -0.75]

def sigmoid(z):
    phi_z = []
    for i in z:
        phi_z.append(1 / (1 + math.exp(-i)))
    return phi_z

""" Test """
#a = 0
#print(sigmoid([a]))
# Should return [0.5]
#print(sigmoid([float('inf')]))
# Should return [1.0]
#print(sigmoid([float('-inf')]))
# Should return [0.0]
#print(sigmoid([-1, 0, 1]))
# Should return [0.2689414213699951, 0.5, 0.7310585786300049]
#print(sigmoid([-100, 100]))
# Should return [3.7200759760208356e-44, 1.0]

def logr_predict_proba(x, w):
    z = net_input(x, w)
    return sigmoid(z)

""" Test """
#x = [[7,4,8],[0,0,2],[7,7,4],[3,0,8]]
#w = [-1, -2, 1]
#print(logr_predict_proba(x, w))
# Should return [0.0009110511944006454, 0.8807970779778823, 4.1399375473943306e-08, 0.9933071490757153]

def logr_predict(x, w):
    y = logr_predict_proba(x, w)
    pred = []
    for i in y:
        if i >= 0.5:
            pred.append(1)
        else:
            pred.append(0)
    return pred

""" Test """
#x = [[7,4,8],[0,0,2],[7,7,4],[3,0,8]]
#w = [-1, -2, 1]
#print(logr_predict(x, w))
# Should return [0, 1, 0, 1]

def logr_cost(x, y, w):
    y_pred = logr_predict_proba(x, w)
    m = len(y)
    cost = 0
    for i in range(m):
        cost += -((1/m) * (y[i] * math.log(y_pred[i]) + ((1-y[i]) * (math.log(1 - y_pred[i])))))
    return cost

""" Test """
#x = [[7,4,8],[0,0,2],[7,7,4],[3,0,8]]
#w = [-1, -2, 1]
#y = [0, 1, 0, 1]
#print(logr_cost(x, y, w))
# Should return 0.033638716846310285

def logr_gradient(x, y, w):
    y_pred = logr_predict_proba(x, w)
    p_grad = 0
    grad = [0 for i in range(len(x))]
    gradient = []
    for i in range(len(x)):
            for j in range(len(x[i])):
                #print("Y%d: %d" % (i, y[i]))
                #print("Pred_Y%d: %.10f" % (i, y_pred[i]))
                #print("X%d, %d: %f" % (i, j, x[i][j]))
                grad[j] += ((y[i] - y_pred[i]) * (-x[i][j]) / len(x))
    return grad
""" WRONG
    x = [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]
    for j in range(len(x)):
        for k in range(len(x[j])):
            grad[j] += (1 / len(x)) * (y[k] - y_pred[k]) * (-x[j][k])
    return grad
"""

""" Test """
#x = [[7,4,8],[0,0,2],[7,7,4],[3,0,8]]
#w = [-1, -2, 1]
#y = [0, 1, 0, 1]
#print(logr_gradient(x, y, w))

def logr_gradient_descent(x, y, w_init, eta, n_iter):
    weights = w_init
    for i in range(n_iter):
        cost = logr_cost(x, y, weights)
        print(cost)
        gradient = logr_gradient(x, y, weights)
        for w in range(len(weights)):
            weights[w] -= (eta * gradient[w])
    return weights

""" Test """
#x = [[7,4,8],[0,0,2],[7,7,4],[3,0,8]]
#y = [0, 1, 0, 1]
#w_init = [0, 0, 0]
#eta = 0.3
#n_iter = 10
#print(logr_gradient_descent(x, y, w_init, eta, n_iter))
"""
x = [[3, 7, 8, 4, 1],
    [3, 0, 7, 9, 5],
    [8, 9, 3, 3, 7],
    [4, 4, 9, 2, 6],
    [4, 4, 5, 3, 6],
    [7, 9, 0, 4, 9],
    [1, 2, 6, 5, 1],
    [6, 8, 8, 2, 8],
    [2, 2, 1, 7, 3],
    [1, 1, 3, 2, 6]]
y = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
w_init = [0, 0, 0, 0, 0]
eta = 0.1
n_iter = 10
print(logr_gradient_descent(x, y, w_init, eta, n_iter))
"""