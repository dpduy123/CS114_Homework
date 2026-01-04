import numpy as np
from sklearn import linear_model


def check_KKT(X, y, bh_vec, n, p, lamda):

    for j in range(p):
        sum = 0
        for i in range(n):
            x_i = X[i, :].reshape((p, 1)).copy()
            sum = sum + (y[i][0] - np.dot(bh_vec.T, x_i)[0][0]) * x_i[j][0]

        s = np.round(sum / (n * lamda), 5)

        print('Feature:', j, 'Sign:', s)

def generate(n, p, true_beta):
    X = np.random.normal(loc=0, scale=1, size=(n, p))
    true_beta = np.reshape(true_beta, (p, 1))

    true_y = np.dot(X, true_beta)
    noise = np.random.normal(loc=0, scale=1, size=(n, 1))
    y = true_y + noise
    return X, y


n = 100
p = 5
lamda = 0.5
beta_vec = [2, -2, 0, 0, 0]

X, y = generate(n, p, beta_vec)

Lasso = linear_model.Lasso(alpha=lamda, fit_intercept=False, tol=1e-10)
Lasso.fit(X, y)

bh = Lasso.coef_
bh_vec = np.reshape(bh, (p, 1))

print(bh)

check_KKT(X, y, bh_vec, n, p, lamda)



