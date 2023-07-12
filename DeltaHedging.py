"""
Attempt at delta hedging

"""


import numpy as np
from numpy import log as ln
from scipy.stats import norm



S = 30 #Stock price
X = 40 #Exercise price
T = 10/365#Time to expiry
sigma = 0.4#standard deviation of log returns (volatility)
r = 0.01 #risk-free interest rate


class BlackScholes:
    
    def __init__(self,S,X,T,sigma,r):
        self.S = S
        self.X = X
        self.T = T
        self.sigma = sigma
        self.r = r


    def blackscholes(self):
        d1 = (ln(self.S/self.X) + self.r+((self.sigma**2)/2)*self.T)/(self.sigma*(self.T**0.5))
        d2 = (ln(self.S/self.X) + self.r-((self.sigma**2)/2)*self.T)/(self.sigma*(self.T**0.5))
        cost = self.S*norm.cdf(d1,0,1)-self.X*np.exp(-self.r*self.T)*norm.cdf(d2,0,1)
        return cost


bs = BlackScholes(S,X,T,sigma,r)
print(bs.blackscholes())

