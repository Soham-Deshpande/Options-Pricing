import numpy as np

def monte_carlo_integration(f, a, b, N, seed=None):
    # f is the function to integrate
    # a and b are the lower and upper limits of integration
    # N is the number of random points to generate
    # seed is an optional random seed to use
    
    rng = np.random.default_rng(seed)
    # Create a new random number generator using the specified seed or a default seed
    
    x = rng.uniform(a, b, size=N)
    # Generate N random points within the interval [a,b]
    
    fx = f(x)
    # Evaluate the function at each random point
    
    integral = (b - a) * np.mean(fx)
    # Approximate the integral as the average of the function values
    # multiplied by the interval width
    
    return integral

def f(x):
  return x**2

a = 0
b = 1
N = 100000

integral = monte_carlo_integration(f, a, b, N)
print("Estimated integral:", integral)
