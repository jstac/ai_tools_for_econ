import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate some sample data points with noise
np.random.seed(42)
x_data = np.linspace(0, 10, 20)
y_true = 2 * np.sin(x_data) + 0.5 * x_data
y_data = y_true + np.random.normal(0, 0.5, size=len(x_data))

# Create a higher resolution x for smooth curve plotting
x_plot = np.linspace(0, 10, 100)

# 1. Polynomial approximation
degrees = [1, 3, 5]  # Linear, cubic, 5th degree
poly_models = []
poly_predictions = []

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_data.reshape(-1, 1), y_data)
    poly_models.append(model)
    poly_predictions.append(model.predict(x_plot.reshape(-1, 1)))

# 2. Custom function approximation (e.g., using a sine function)
def custom_func(x, a, b, c, d):
    return a * np.sin(b * x) + c * x + d

# Fit the custom function
params, _ = curve_fit(custom_func, x_data, y_data, p0=[1, 1, 0.1, 0])
custom_prediction = custom_func(x_plot, *params)

# Create the visualization
plt.figure(figsize=(12, 8))

# Plot original data points
plt.scatter(x_data, y_data, color='black', label='Data points')

# Plot true function (for comparison)
plt.plot(x_plot, 2 * np.sin(x_plot) + 0.5 * x_plot, 'k--', label='True function')

# Plot polynomial approximations
colors = ['blue', 'green', 'red']
for i, degree in enumerate(degrees):
    plt.plot(x_plot, poly_predictions[i], color=colors[i], 
             label=f'Polynomial (degree={degree})')

# Plot custom function approximation
plt.plot(x_plot, custom_prediction, color='purple', 
         label=f'Custom function: {params[0]:.2f}*sin({params[1]:.2f}*x) + {params[2]:.2f}*x + {params[3]:.2f}')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Approximation from Data Points')
plt.grid(True, alpha=0.3)

# Add RMSE values as text
for i, degree in enumerate(degrees):
    rmse = np.sqrt(np.mean((poly_predictions[i] - (2 * np.sin(x_plot) + 0.5 * x_plot))**2))
    plt.text(0.05, 0.9 - i*0.05, f'RMSE Poly(degree={degree}): {rmse:.4f}', 
             transform=plt.gca().transAxes, color=colors[i])

custom_rmse = np.sqrt(np.mean((custom_prediction - (2 * np.sin(x_plot) + 0.5 * x_plot))**2))
plt.text(0.05, 0.75, f'RMSE Custom: {custom_rmse:.4f}', 
         transform=plt.gca().transAxes, color='purple')

plt.tight_layout()
plt.savefig('function_approximation.png', dpi=300)
plt.show()
