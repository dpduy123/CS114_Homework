import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Create example data
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, loc=0, scale=1)
cdf = norm.cdf(x, loc=0, scale=1)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot PDF
plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', label='PDF')
plt.title('Probability Density Function')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Plot CDF
plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'r-', label='CDF')
plt.title('Cumulative Distribution Function')
plt.xlabel('x')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()

# Example point
T = 1.5
cdf_value = norm.cdf(T, loc=0, scale=1)
print(f"For T = {T}:")
print(f"CDF value P(X ≤ T) = {cdf_value:.4f}")
print(f"Two-sided p-value = {2 * min(cdf_value, 1-cdf_value):.4f}")

# Mark the point on CDF plot
plt.subplot(1, 2, 2)
plt.plot([T], [cdf_value], 'go', label=f'T={T}')
plt.legend()

plt.tight_layout()
plt.savefig('cdf_visualization.png')
plt.close() 