import numpy as np
from scipy.stats import norm

def black_scholes(S0, K, T, r, sigma):
  """
  Calculates the price of a European call option using the Black-Scholes model.

  Args:
    S0: The current price of the underlying asset.
    K: The strike price of the option.
    T: The time to maturity of the option, in years.
    r: The risk-free interest rate.
    sigma: The volatility of the underlying asset.

  Returns:
    The price of the option.
  """

  d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)
  d2 = d1 - sigma * np.sqrt(T)

  return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
