import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Example 1: 2D multivariate normal with mean [1,2] and identity covariance matrix
mean = [1, 2]
cov = [[1, 0], [0, 1]]  # identity matrix
samples = np.random.multivariate_normal(mean, cov, size=5)

print("Example 1: 2D multivariate normal distribution")
print("Mean vector:", mean)
print("Covariance matrix:\n", cov)
print("5 random samples:\n", samples)

# Example 2: 1D case (like in your code)
n = 10
mean_1d = np.ones(n) * 2  # array of 2's
cov_1d = np.identity(n)   # identity matrix
samples_1d = np.random.multivariate_normal(mean_1d, cov_1d)

print("\nExample 2: 1D case (similar to your code)")
print("First few values of the sample:", samples_1d[:5]) 