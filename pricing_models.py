import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the theoretical price of a European Call Option.
    
    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to expiration (in years)
    r : float : Risk-free interest rate (e.g., 0.05 for 5%)
    sigma : float : Volatility (e.g., 0.2 for 20%)
    """
    
    # Calculate d1 and d2 (intermediate variables for the Black-Scholes formula)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate the Call Price using the Cumulative Distribution Function (norm.cdf)
    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    
    return call_price