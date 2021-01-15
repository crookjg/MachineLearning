import numpy as np
import matplotlib.pyplot as plt

def net_input_numpy(X, w, w0):
    vec = []
    for u in X:
        prod = w0
        for ui, vi in zip(u, w):
            prod += ui * vi
        vec.append(prod)
    return np.array(vec)

def svm_w0_gradient(x, y, w, w0, lambda_):
    der_w0 = 0
    for i in range(len(y)):
        der_w0 += y[i] * lambda_[i]

    return (-1 * der_w0)

def svm_w_gradient(x, y, w, w0, lambda_):
    x_l = []
    for i in range(len(w)):
        x_l_temp = 0
        for j in range(len(y)):
            x_l_temp += (lambda_[j] * y[j] * x[j][i])
        x_l.append(w[i] - x_l_temp)
    return x_l


def svm_lambda_gradient(x, y, w, w0, lambda_):
    z = net_input_numpy(x, w, w0)
    u = []
    for i in range(len(y)):
        u.append(-1 * (y[i] * z[i] - 1))
    
    return np.array(u)


def svm_active_lambda_gradient(lambda_, lambda_gradient, c=np.inf):
    active = []
    for i in range(len(lambda_)):
        if (lambda_gradient[i] >= (-1 * lambda_[i])) and (lambda_gradient[i] <= (c - lambda_[i])):
            active.append(lambda_gradient[i])
        elif (lambda_gradient[i] < (-1 * lambda_[i])):
            active.append(-1 * lambda_[i])
        else:
            active.append(c - lambda_[i])

    return np.array(active)

def svm_support_vectors(lambda_, c=np.inf):
    tf = [False for l in lambda_]
    for i in range(len(lambda_)):
        if (lambda_[i] > 0) and (lambda_[i] <= c):
            tf[i] = True
    return np.array(tf)

def svm_on_correct_hyperplane(lambda_, c=np.inf):
    tf = [False for l in lambda_]
    for i in range(len(lambda_)):
        if (lambda_[i] > 0) and (lambda_[i] < c):
            tf[i] = True
    return np.array(tf)

def svm_ksi(x, y, w, w0):
    z = net_input_numpy(x, w, w0)
    ksi = []
    for i in range(len(y)):
        ksi_temp = 1 - y[i] * z[i]
        if ksi_temp > 0:
            ksi.append(ksi_temp)
        else:
            ksi.append(0)
    
    return np.array(ksi)

def svm_misclassified(ksi):
    return np.array(ksi > 1.)

def svm_optimization(x, y, c, eta, n_iter):
    w = [0 for j in x[0]]
    w0 = 0
    lambda_ = [1 for i in y]
    for i in range(n_iter):
        # Compute gradients
        w0_grad = svm_w0_gradient(x, y, w, w0, lambda_)
        w_grad = svm_w_gradient(x, y, w, w0, lambda_)
        l_grad = svm_lambda_gradient(x, y, w, w0, lambda_)
        l_act = svm_active_lambda_gradient(lambda_, l_grad, c)

        print("epoch %d: max(abs(dL/dwj)) = %.3e, max(abs(dL/d lambda)) = %.3e" % (i + 1, np.abs(w_grad).max(), np.abs(l_act).max()))
        # Update w0, w, and lambda
        w0 -= (eta * w0_grad)
        for k in range(len(w)):
            w[k] += (-1 * (eta * w_grad[k]))
        for j in range(len(l_grad)):
            lambda_[j] += (eta * l_grad[j])
            # Make sure lambda is >= 0 and <= c
            if lambda_[j] > c:
                lambda_[j] = c
            elif lambda_[j] < 0:
                lambda_[j] = 0

    # Print # of misclassifications
    ksi = svm_ksi(x, y, w, w0)
    mis = svm_misclassified(ksi)
    mis_clas = 0
    for b in mis:
        if b == True:
            mis_clas += 1
    print("%d misclassified" % mis_clas)

    # Print # of support vectors
    sup_vec = svm_support_vectors(lambda_, c)
    sv = 0
    for b in sup_vec:
        if b == True:
            sv += 1
    print("%d support vectors" % sv)

    # Print # of samples on correct hyperplane
    hyp = svm_on_correct_hyperplane(lambda_, c)
    h_c = 0
    for b in hyp:
        if b == True:
            h_c += 1
    print("%d on correct hyperplane" % h_c)

    # Print max of either partial derivative of weight or lambda active
    w_grad_fin = svm_w_gradient(x, y, w, w0, lambda_)
    n_l_grad = svm_lambda_gradient(x, y, w, w0, lambda_)
    l_act_fin = svm_active_lambda_gradient(lambda_, n_l_grad, c)
    if (np.abs(w_grad_fin).max() > np.abs(l_act_fin).max()):
        print("max(abs(active(gradient))) = %e" % np.abs(w_grad_fin).max())
    else:
        print("max(abs(active(gradient))) = %e" % np.abs(l_act_fin).max())
    return w, w0, lambda_

def svm_plot(x, y, c, w, w0, lambda_):
    fig, ax = plt.subplots()
    index = y == 1
    ax.scatter(x[index, 0], x[index, 1], marker='+', c='tab:orange', edgecolors='black',label='positive')
    index = y == -1
    ax.scatter(x[index, 0], x[index, 1], marker='o', c='blue', edgecolors='black',label='negative')
    index = svm_support_vectors(lambda_, c)
    ax.scatter(x[index, 0], x[index, 1], marker='s', c='none', s=100, edgecolors='black',label='support vectors')
    index = svm_on_correct_hyperplane(lambda_, c)
    ax.scatter(x[index, 0], x[index, 1], marker='o', c='none', s=100, edgecolors='black',label='on correct hyperplane')
    index = svm_misclassified(svm_ksi(x, y, w, w0))
    ax.scatter(x[index, 0], x[index, 1], marker='d', c='none', s=100, edgecolors='black',label='misclassified')

    xx1, xx2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    zz1 = w0 + w[0] * xx1 + w[1] * xx2
    ax.contour(xx1, xx2, zz1, levels=[-1, 0, 1],linestyles=['dotted', 'dashed', 'dotted'], colors='black')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(f'C = {c}')
    fig.legend()
    plt.show()


""" Test """
#x = np.array([[-4, 2, 3],[-1, 0, 2],[-4, 6, -9],[7, -8, 3]])
#w = np.array([-0.2, 0.4, -0.3])
#w0 = 0.9
#y = np.array([1,  1,  1, -1])
#l = np.array([0.4, 0.5, 0.2, 0.4])
#print(net_input_numpy(x, w, w0))
#print(svm_w0_gradient(x, y, w, w0, l))
#print(svm_w_gradient(x, y, w, w0, l))
#print(svm_lambda_gradient(x, y, w, w0, l))

#l_n = np.array([0.5, 0.8, 0.9, 0.5, 0. , 0. , 0.1, 0.7, 0.6, 0.9])
#l_g = np.array([ 0.8,  1. , -0.5,  0.8,  1. ,  0.1,  0. ,  0.4,  0.8, -0.6])
#print(svm_active_lambda_gradient(l_n, l_g, 1))

#n_l = np.array([0, 10, 0, 6.1, 0, 3.9, 0, 10])
#print(svm_support_vectors(n_l, 10))

#print(svm_ksi(x, y, w, w0))

#ksi = np.array([4. , 1.5, 7.2, 2.2, 4.3])
#print(svm_misclassified(ksi))

"""
np.random.seed(2)
n, m = 20, 2
x = np.random.uniform(-3, 3, (n, m))
w = np.random.normal(0, 3, (m,))
z = x.dot(w)
eta = 0.05
n_iter = 1000
# Linearly separable case
y = (z >= 0).astype(int) * 2 - 1
# what should happen when c=0?
c = float('inf')
w, w0, lambda_ = svm_optimization(x, y, c, eta, n_iter)

svm_plot(x, y, c, w, w0, lambda_)
"""