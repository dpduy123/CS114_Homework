import numpy as np
from scipy.stats import norm


def run():
    # generate synthetic data
    n = 50
    m = 100

    mu_X = np.ones((n, 1)) * 2
    mu_Y = np.ones((m, 1)) * 2

    Sigma_X = np.identity(n)
    Sigma_Y = np.identity(m)

    X = np.random.multivariate_normal(mu_X.flatten(), Sigma_X)
    X = X.reshape((n, 1))

    Y = np.random.multivariate_normal(mu_Y.flatten(), Sigma_Y)
    Y = Y.reshape((m, 1))

    # compute test statistic
    T = np.mean(X) - np.mean(Y)

    # compute standard deviation of the distribution of T
    sigma_T = np.sqrt(1/n + 1/m)

    # distribution of T ~ N(mu_X - mu_Y, sigma_T)

    # compute two-sided p-value
    cdf = norm.cdf(T, loc=0, scale=sigma_T)
    p_value = 2 * min(cdf, 1 - cdf)

    return p_value


if __name__ == '__main__':
    # Exercise 1: Your code to compute p_value
    max_iteration = 10000
    list_p_value = []
    alpha = 0.5
    count = 0

    for _ in range(max_iteration):
        p_value = run()
        list_p_value.append(p_value)

        if p_value <= alpha:
            count = count + 1
    print("False Positive Rate: ", count / max_iteration)