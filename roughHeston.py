import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy.special import gamma
from scipy.integrate import quad, trapezoid
import pandas as pd


class roughHeston:
    
    def __init__(self, nbTimeSteps, heston_params, T):
        # Time discretisation parameters
        self.T = T
        self.n = nbTimeSteps
        self.dt = self.T / self.n
        self.time_grid = np.linspace(0., T, self.n + 1)

        # Heston model paramters
        self.S0 = heston_params['S0']
        self.kappa = heston_params['kappa']
        self.nu = heston_params['nu']
        self.theta = heston_params['theta']
        self.alpha = heston_params['alpha']
        self.V0 = heston_params['V0']
        self.rho = heston_params['rho']

        # Precomputations to speed up pricing
        self.frac = self.dt**self.alpha / gamma(self.alpha + 2.)
        self.frac2 = self.dt**self.alpha / gamma(self.alpha + 1.)
        self.frac_bar = 1. / gamma(1.-self.alpha)
        self.fill_a()
        self.fill_b()

    # Fractional Riccati equation
    def F(self, a, x):
        return -0.5*(a*a + 1j *a) - (self.kappa - 1j*a*self.rho*self.nu)*x + 0.5*self.nu*self.nu*x*x

    # Filling the coefficient a and b which don't depend on the characteristic function
    def a(self, j, k):
        if j == 0:
            res = ((k - 1)**(self.alpha + 1) - (k - self.alpha - 1)*k**self.alpha)
        elif j == k:
            res = 1.
        else:
            res = ((k + 1 - j)**(self.alpha + 1) + (k - 1 - j)**(self.alpha + 1) - 2 * (k - j)**(self.alpha + 1))

        return self.frac*res

    def fill_a(self):
        self.a_ = np.zeros(shape = (self.n + 1, self.n + 1))
        for k in range(1, self.n + 1):
            for j in range(k + 1):
                self.a_[j, k] = self.a(j, k)

    def b(self, j, k):
        return self.frac2*((k - j)**self.alpha - (k - j - 1)**self.alpha)

    def fill_b(self):
        self.b_ = np.zeros(shape = (self.n, self.n + 1))
        for k in range(1, self.n + 1):
            for j in range(k):
                self.b_[j, k] = self.b(j, k)

    # Computation of two sums used in the scheme
    def h_P(self, a, k):
        res = 0
        for j in range(k):
            res += self.b_[j, k] * self.F(a, self.h_hat[j])
        return res

    def sum_a(self, a, k):
        res = 0
        for j in range(k):
            res += self.a_[j, k] * self.F(a, self.h_hat[j])
        return res

    # Solving function h for each time step
    def fill_h(self, a):
        self.h_hat = np.zeros((self.n + 1), dtype=complex)
        for k in range(1, self.n + 1):
            h_P = self.h_P(a, k)
            sum_a = self.sum_a(a, k)
            self.h_hat[k] = sum_a + self.a_[k, k]*self.F(a, h_P)

    # Characteristic function computation
    def rHeston_char_function(self, a):
        # Filling the h function
        self.fill_h(a)

        # Standard integral of the h function
        integral = trapezoid(self.h_hat, self.time_grid)

        # Fractional integral of the h function
        func = lambda s: (self.T - s)**(1. - self.alpha)
        y = np.fromiter((((func(self.time_grid[i]) - func(self.time_grid[i+1]))*self.h_hat[i]) for i in range(self.n)), self.h_hat.dtype)
        frac_integral = self.frac_bar * np.sum(y) / (1.-self.alpha)

        # Characteristic function
        return np.exp(self.kappa*self.theta*integral + self.V0*frac_integral)

    # Pricing with an inverse Fourier transform
    def rHeston_Call(self, k, upLim):
        K = self.S0*np.exp(k)
        func = lambda u: np.real(np.exp(-1j*u*k)*self.rHeston_char_function(u-0.5*1j)) / (u**2 + 0.25)
        integ = quad(func, 0, 5.)
        return self.S0 - np.sqrt(self.S0*K) * integ[0] / np.pi

    # Analytical formula for the standard Heston characteristic function
    def heston_char_function(self,u):
        nu2 = self.nu**2
        T = self.T
        dif = self.kappa - self.rho*self.nu*u*1j
        d = np.sqrt(dif**2 + nu2 *(1j*u + u**2))
        g = (dif - d) / (dif + d)
        return np.exp(1j*u*(np.log(self.S0)))\
               *np.exp((self.kappa*self.theta/nu2) * ((dif-d)*T - 2.*np.log((1. - g*np.exp(-d*T))/(1.-g))))\
               *np.exp((self.V0/nu2) * (dif-d)*(1.-np.exp(-d*T))/(1-g*np.exp(-d*T)))

    # Pricing with an inverse Fourier transform
    def heston_Call(self, k):
        K = self.S0 * np.exp(k)
        func = lambda u: np.real(np.exp(-1j*u*k) * self.heston_char_function(u-0.5*1j)) / (u**2+0.25)
        integ = quad(func, 0, np.inf)
        return self.S0 - np.sqrt(self.S0*K) * integ[0] / np.pi


def BlackScholesCallPut(S, K, T, sigma, r, call_put=1):
    d1 = (np.log(S/K) + (r+.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return call_put*(S*norm.cdf(call_put*d1) - K*np.exp (-r*T) * norm.cdf (call_put*d2))

def impliedVol(S, K, T, r, price):
    def smileMin(vol, *args):
        S, K, T, r, price = args
        return price - BlackScholesCallPut(S, K, T, vol, r, 1)
    vMin = 0.0001
    vMax = 3.
    return bisect(smileMin, vMin, vMax, args=(S, K, T, r, price), rtol=1e-15, full_output=False, disp=True)
# Heston parameters
kappa = .3
nu = .3
rho = -.7
V0 = .02
theta = .02
S0 = 1.
T = 1.

alpha=0.6

nbTimeSteps, upLim = 100, 5.

heston_params = {'kappa': kappa, 'nu': nu, 'alpha': alpha, 'rho': rho, 'V0': V0, 'theta': theta, 'S0': S0}
logmoneyness = -0.1
K = S0*np.exp(logmoneyness)
he = roughHeston(nbTimeSteps, heston_params, T)
p = he.heston_Call(logmoneyness)
rp = he.rHeston_Call(logmoneyness, upLim)
riv = impliedVol(S0, K, T, 0., rp)
iv = impliedVol(S0, K, T, 0., p)

nbDec = 4
print("Heston option price and implied volatility:", np.round(p,nbDec), np.round(iv,nbDec), np.round(BlackScholesCallPut(S0,K,T,iv,0.,1), nbDec))
print("rough Heston option price and implied volatility:", np.round(rp,nbDec), np.round(riv,nbDec), np.round(BlackScholesCallPut(S0,K,T,riv,0.,1), nbDec))
data = {'Heston':  [0.2, 0.3],
        'rough Heston': [0.1, 0.2]}

df = pd.DataFrame(data, index=["Call Price", "implied vol"])
print(df)
