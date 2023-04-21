import numpy as np

def rk4(f, y0, t0, tf, h):
    # Define the time grid
    t = np.arange(t0, tf+h, h)
    n = len(t)
    
    # Initialize the solution array
    y = np.zeros(n)
    y[0] = y0
    
    # Apply the Runge-Kutta method
    for i in range(n-1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2*k1)
        k3 = f(t[i] + h/2, y[i] + h/2*k2)
        k4 = f(t[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

def f(t, y):
  return -2*t*y



t, y = rk4(f, y0=1, t0=0, tf=1, h=0.1)

print("t:", t)
print("y:", y)
