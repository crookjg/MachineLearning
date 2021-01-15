def dot_product(u, v):
    prod = 0
    for ui, vi in zip(u,v):
        prod += ui * vi
    return prod

def matrix_multiply(a, b):
    return [[sum(ai * bi for ai, bi, in zip(a_row, b_col)) for b_col in zip(*b)] for a_row in a]

""" Simple Linear Regression Using Cost Function """
def slr_cost(x, y, w_0, w_1):
    sum = 0
    for i in range(len(x)):
        sum += (y[i] - w_0 - (w_1 * x[i])) ** 2
    cost = sum / (2 * len(x))
    return cost

#x = [8, 8, 4, 3, 6, 9, 8, 8, 3, 1]
#w_0 = 2
#w_1 = 0.1
#y = [3.90, 3.87, 3.10, 2.95, 3.64, 4.26, 4.04, 4.04, 2.57, 2.15]
#print(slr_cost(x, y, w_0, w_1))

""" Simple Linear Regression Gradient """
def slr_gradient(x, y, w_0, w_1):
    w_0_sum = 0
    w_1_sum = 0
    for i in range(len(x)):
        w_0_sum += ((y[i] - w_0 - (w_1 * x[i])) * -1)
        w_1_sum += ((y[i] - w_0 - (w_1 * x[i])) * (-1 * x[i]))
    grad_w_0 = w_0_sum / len(x)
    grad_w_1 = w_1_sum / len(x)
    return (grad_w_0, grad_w_1)

#x = [8, 8, 4, 3, 6, 9, 8, 8, 3, 1]
#w_0 = 2
#w_1 = 0.1
#y = [3.90, 3.87, 3.10, 2.95, 3.64, 4.26, 4.04, 4.04, 2.57, 2.15]
#print(slr_gradient(x, y, w_0, w_1))

""" Simple Linear Regression Using Least Squares """
def slr_analytical(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    w_1_top = 0
    w_1_bott = 0
    for i in range(len(x)):
        half = x[i] - mean_x
        w_1_top += half * (y[i] - mean_y)
        w_1_bott += half ** 2
    w_1 = w_1_top / w_1_bott
    w_0 = mean_y - (w_1 * mean_x)
    return (w_0, w_1)

#x = [8, 8, 4, 3, 6, 9, 8, 8, 3, 1]
#y = [3.90, 3.87, 3.10, 2.95, 3.64, 4.26, 4.04, 4.04, 2.57, 2.15]
#print(slr_analytical(x, y))

def slr_gradient_descent(x, y, w_0_init, w_1_init, eta, n_iter):
    w_0 = w_0_init
    w_1 = w_1_init

    for i in range(n_iter):
        print(slr_cost(x, y, w_0, w_1))
        w_0_temp, w_1_temp = slr_gradient(x, y, w_0, w_1)
        w_0 -= eta * w_0_temp
        w_1 -= eta * w_1_temp
    return w_0, w_1

#w0_init, w1_init = 0, 0
#eta = 0.03
#n_iter = 10
#x = [8, 8, 4, 3, 6, 9, 8, 8, 3, 1]
#y = [3.90, 3.87, 3.10, 2.95, 3.64, 4.26, 4.04, 4.04, 2.57, 2.15]
#print(slr_gradient_descent(x, y, w0_init, w1_init, eta, n_iter))