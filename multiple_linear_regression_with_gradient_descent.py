import numpy as np
from sklearn.linear_model import LinearRegression


def generate_data(n, p, true_beta):
    X = np.random.normal(loc=0, scale=1, size=(n, p))

    true_beta = np.reshape(true_beta, (p, 1))

    noise = np.random.normal(loc=0, scale=1, size=(n, 1))
    y = np.dot(X, true_beta) + noise

    return X, y


n = 40
p = 5
true_beta = [-1.0, 0.0, 1.0, 1.5, 2.0]

generate_data(n, p, true_beta)
X, y = generate_data(n, p, true_beta)

# Exercise 1: regression with sklearn

reg = LinearRegression(fit_intercept=False)
reg.fit(X, y.flatten())

print()
print('With sklearn:', reg.coef_)
print()

# Exercise 2: regression without sklearn

XTX = np.dot(X.T, X)
XTXinv = np.linalg.inv(XTX)
XTXinvXT = np.dot(XTXinv, X.T)
beta = np.dot(XTXinvXT, y)

print('Without sklearn:', beta.flatten())
print()


# Exercise 3: regression with gradient descent


def compute_gradient(j, beta):
    gradient = 0
    for i in range(n):
        xi = X[i, :].reshape((p, 1))
        xi_j = xi[j][0]
        gradient = gradient + xi_j * (y[i][0] - np.dot(beta.T, xi)[0][0])
    gradient = - gradient
    return gradient

def gradient_descent(beta_init, alpha):
    list_beta = [beta_init]
    for iter in range(1000):
        beta_new = list_beta[-1].copy()
        for j in range(p):
            gradient = compute_gradient(j, beta_new)
            beta_new[j][0] = beta_new[j][0] - alpha * gradient
        list_beta.append(beta_new)
    return list_beta[-1]

alpha = 0.01
beta_init = np.random.normal(loc=0, scale=1, size=(p, 1))
beta = gradient_descent(beta_init, alpha)
print('Gradient Descent: ', beta.flatten())
