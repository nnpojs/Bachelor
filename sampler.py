import scipy.optimize as opt
import scipy
from scipy.integrate import quad
import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
def fun(x, d):
    # Calculate the incomplete gamma function Gamma(d/2, 2 * pi^2 * x^2)
    gamma_val = sp.gammaincc(d / 2, 2 * (np.pi**2) * (x**2)) * sp.gamma(d / 2)
    a = (d * np.pi * (2 * (np.pi**2))**((d-1)/2)) / (np.sqrt(2) * sp.gamma((d+2)/2))
    # Calculate the expression
    function = -(2**(-d / 2 - 1) * gamma_val * x) / (np.pi**d * np.abs(x)) * a
    if x > 0:
        function += 1
    return function

def invert_fun(y, d, x_min=-100, x_max=100):
    # Define the objective function: f(x) - y
    def objective(x):
        return fun(x, d) - y

    # Use root_scalar to solve f(x) = y
    sol = opt.root_scalar(objective, bracket=[x_min, x_max], method='brentq')
    
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Root finding did not converge")



# Find the inverse
#try:
#    x_inverted = invert_fun(y_target, d)
#    print(f"Inverse found: x = {x_inverted}")
#except ValueError as e:
#    print(e)

def sample(d,N):
    ys=np.random.uniform(low=0.0, high=1.0, size=N)
    xs=np.zeros(N)
    for i in range(N):
        xs[i] = invert_fun(ys[i], d)
    laba=[]
    '''
    for i in xs:
        if i>0:
            laba.append(i)
    print(np.mean(laba))
    '''
    return xs

def sample_sphere(d, num_samples=1):
    # Generate num_samples points from a multivariate normal distribution
    random_points = np.random.randn(num_samples, d)
    
    # Normalize each point to have unit length (project onto the sphere)
    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
    sphere_points = random_points / norms
    
    return sphere_points
'''
d=50
N=10000
# Plotting the x-coordinates
plt.figure(figsize=(8, 5))
plt.plot(sample(d,N), 'o', color='blue',markersize=1)  # Plot points as markers
plt.title(f"Sampled {N} Points (X-values)")
plt.xlabel("Index")
plt.ylabel("Value on [-1, 1]")
plt.grid(True, linestyle='--', alpha=0.6)


# Display the plot
plt.show()
'''
# Define the range of x values and parameter d
x_vals = np.linspace(-2, 2, 400)  # Change this range as needed
d = 50  # Example value for d, adjust as required

# Compute the y values for the given x values
y_vals = [fun(x, d) for x in x_vals]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=f'CDF for d={d}')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of CDF')
plt.legend()
plt.grid(True)
plt.show()

