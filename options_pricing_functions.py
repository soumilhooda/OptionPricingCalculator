import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from datetime import datetime as dt
from scipy import stats
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from math import log, sqrt, exp

mpl.rcParams['font.family'] = 'serif'

# Model Parameters
S0 = 100.0  # index level
K = 100.0  # option strike
T = 1.0  # maturity date
r = 0.05  # risk-less short rate
sigma = 0.2  # volatility

def dN(x):
    ''' Probability density function of standard normal random variable x. '''
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def N(d):
    ''' Cumulative density function of standard normal random variable x. '''
    return quad(lambda x: dN(x), -20, d, limit=50)[0]


def d1f(St, K, t, T, r, sigma):
    ''' Black-Scholes-Merton d1 function.
        Parameters see e.g. BSM_call_value function. '''
    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2)
          * (T - t)) / (sigma * math.sqrt(T - t))
    return d1


def BSM_call_value(St, K, t, T, r, sigma):
    ''' Calculates Black-Scholes-Merton European call option value.

    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        date of maturity/time-to-maturity if t = 0; T > t
    r : float
        constant, risk-less short rate
    sigma : float
        volatility

    Returns
    =======
    call_value : float
        European call present value at t
    '''
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    call_value = St * N(d1) - math.exp(-r * (T - t)) * K * N(d2)
    return call_value


def BSM_put_value(St, K, t, T, r, sigma):
    ''' Calculates Black-Scholes-Merton European put option value.

    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        date of maturity/time-to-maturity if t = 0; T > t
    r : float
        constant, risk-less short rate
    sigma : float
        volatility

    Returns
    =======
    put_value : float
        European put present value at t
    '''
    put_value = BSM_call_value(St, K, t, T, r, sigma) \
        - St + math.exp(-r * (T - t)) * K
    return put_value


def BSM_option_value(S0, K, T, r, sigma, otype):
    """Calculates Black-Scholes-Merton European option value."""
    if otype == 'call':
        return BSM_call_value(S0, K, 0, T, r, sigma)  # t = 0 for European options
    elif otype == 'put':
        return BSM_put_value(S0, K, 0, T, r, sigma)
    else:
        raise ValueError("Invalid option type.")

def CRR_option_value(S0, K, T, r, sigma, otype, M=4):
    ''' Cox-Ross-Rubinstein European option valuation.

    Parameters
    ==========
    S0 : float
        stock/index level at time 0
    K : float
        strike price
    T : float
        date of maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    otype : string
        either 'call' or 'put'
    M : int
        number of time intervals
    '''
    # Time Parameters
    dt = T / M  # length of time interval
    df = math.exp(-r * dt)  # discount per interval

    # Binomial Parameters
    u = math.exp(sigma * math.sqrt(dt))  # up movement
    d = 1 / u  # down movement
    q = (math.exp(r * dt) - d) / (u - d)  # martingale branch probability

    # Array Initialization for Index Levels
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Inner Values
    if otype == 'call':
        V = np.maximum(S - K, 0)  # inner values for European call option
    else:
        V = np.maximum(K - S, 0)  # inner values for European put option

    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        V[0:M - z, t] = (q * V[0:M - z, t + 1] +
                         (1 - q) * V[1:M - z + 1, t + 1]) * df
        z += 1
    return V[0, 0]



def Jarrow_Rudd_option_value(S0, K, T, r, sigma, otype, M=4):
    ''' Jarrow and Rudd's modified Cox-Ross-Rubinstein European option valuation.

    Parameters
    ==========
    S0 : float
        stock/index level at time 0
    K : float
        strike price
    T : float
        date of maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    otype : string
        either 'call' or 'put'
    M : int
        number of time intervals
    '''
    # Time Parameters
    dt = T / M  # length of time interval
    df = math.exp(-r * dt)  # discount per interval

    # Binomial Parameters
    u = math.exp((r-(sigma**2)/2)*dt + sigma * math.sqrt(dt))  # up movement
    d = math.exp((r-(sigma**2)/2)*dt - sigma * math.sqrt(dt))  # down movement
    q = (math.exp(r * dt) - d) / (u - d)  # martingale branch probability

    # Array Initialization for Index Levels
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Inner Values
    if otype == 'call':
        V = np.maximum(S - K, 0)  # inner values for European call option
    else:
        V = np.maximum(K - S, 0)  # inner values for European put option

    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        V[0:M - z, t] = (q * V[0:M - z, t + 1] +
                         (1 - q) * V[1:M - z + 1, t + 1]) * df
        z += 1
    return V[0, 0]


def Tian_option_value(S0, K, T, r, sigma, otype, M=4):
    ''' Tian's modfied Cox-Ross-Rubinstein European option valuation.

    Parameters
    ==========
    S0 : float
        stock/index level at time 0
    K : float
        strike price
    T : float
        date of maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    otype : string
        either 'call' or 'put'
    M : int
        number of time intervals
    '''
    # Time Parameters
    dt = T / M  # length of time interval
    df = math.exp(-r * dt)  # discount per interval

    # Binomial Parameters
    v = math.exp((sigma**2) * dt)
    u = 0.5 * math.exp(r * dt) * v * (v + 1 + math.sqrt(v**2 + 2*v -3))  # up movement
    d = 0.5 * math.exp(r * dt) * v * (v + 1 - math.sqrt(v**2 + 2*v -3))  # down movement
    q = (math.exp(r * dt) - d) / (u - d)  # martingale branch probability

    # Array Initialization for Index Levels
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Inner Values
    if otype == 'call':
        V = np.maximum(S - K, 0)  # inner values for European call option
    else:
        V = np.maximum(K - S, 0)  # inner values for European put option

    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        V[0:M - z, t] = (q * V[0:M - z, t + 1] +
                         (1 - q) * V[1:M - z + 1, t + 1]) * df
        z += 1
    return V[0, 0]


def trinomial_option_value(S0, K, T, r, sigma, otype, M=4):
    """Trinomial tree option valuation."""
    dt = T / M
    df = math.exp(-r * dt)

    # Trinomial parameters
    u = math.exp(sigma * math.sqrt(2 * dt))
    d = 1 / u
    m = 1  # Middle movement

    # Probabilities
    pu = ((math.exp(r * dt) - d) / (u - d)) ** 2
    pd = ((u - math.exp(r * dt)) / (u - d)) ** 2
    pm = 1 - pu - pd

    # Stock price lattice
    S = np.zeros((M + 1, M + 1))
    for i in range(M + 1):
        for j in range(M + 1):
            S[i, j] = S0 * u ** (i - j) * d ** j

    # Option values at maturity
    if otype == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)

    # Backward iteration
    for t in range(M - 1, -1, -1):
        for i in range(t + 1):
            V[i, t] = df * (pu * V[i + 1, t + 1] + pm * V[i, t + 1] + pd * V[i - 1, t + 1])

    return V[0, 0]

# Monte Carlo Simulation
def monte_carlo_option_value(S0, K, T, r, sigma, otype, num_sims=10000):
    """Monte Carlo simulation for European option valuation."""
    dt = T / num_sims
    S_paths = np.zeros((num_sims + 1, num_sims))
    S_paths[0] = S0

    # Generate random paths
    for t in range(1, num_sims + 1):
        Z = np.random.normal(0, 1, num_sims)
        S_paths[t] = S_paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * Z)

    # Calculate option values at maturity
    if otype == 'call':
        V = np.maximum(S_paths[-1] - K, 0)
    else:
        V = np.maximum(K - S_paths[-1], 0)

    # Discount and average
    option_value = math.exp(-r * T) * np.mean(V)
    return option_value

# Modified Ornstein-Uhlenbeck Process (MOU)
def MOU_option_value(S0, K, T, r, sigma, otype, kappa, theta, M=4):
    """Option valuation using MOU process."""
    dt = T / M
    df = math.exp(-r * dt)

    # MOU parameters
    alpha = kappa * theta
    beta = kappa

    # Stock price lattice
    S = np.zeros((M + 1, M + 1))
    S[0] = S0
    for t in range(1, M + 1):
        Z = np.random.normal(0, 1, M + 1)
        S[t] = S[t - 1] * np.exp((r - alpha - 0.5 * sigma ** 2) * dt + 
                                 sigma * math.sqrt(dt) * Z + 
                                 beta * (S[t - 1] - alpha) * dt)

    # Option values at maturity
    if otype == 'call':
        V = np.maximum(S[-1] - K, 0)
    else:
        V = np.maximum(K - S[-1], 0)

    # Backward iteration
    for t in range(M - 1, -1, -1):
        for i in range(t + 1):
            V[i, t] = df * np.mean(V[i + 1, t + 1]) 

    return V[0, 0]

def calculate_greeks(S0, K, T, r, sigma, otype):
    """Calculates the Greeks for a European option."""
    d1 = d1f(S0, K, 0, T, r, sigma)  # t = 0 for European options
    d2 = d1 - sigma * math.sqrt(T)

    if otype == 'call':
        delta = N(d1)
        gamma = dN(d1) / (S0 * sigma * math.sqrt(T))
        vega = S0 * dN(d1) * math.sqrt(T)
        theta = -(S0 * dN(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N(d2)
        rho = K * T * math.exp(-r * T) * N(d2)
    elif otype == 'put':
        delta = N(d1) - 1
        gamma = dN(d1) / (S0 * sigma * math.sqrt(T))
        vega = S0 * dN(d1) * math.sqrt(T)
        theta = -(S0 * dN(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * N(-d2)
        rho = -K * T * math.exp(-r * T) * N(-d2)
    else:
        raise ValueError("Invalid option type.")

    return delta, gamma, vega, theta, rho

# European Option Class
class european_option:
    
    def __init__(self, S0, K, t, M, r, sigma, d, CP, C=10):
        
        '''
        Attributes:
        ------------------------------------------------------------------------------------------
        S0 (initial underlying level), 
        K (option strike price), 
        t (pricing date),                  ## Can enter in datetime format or str ('dd/mm/yyyy') or years (float)
        M (maturity date),                 ## Same as t
        r (constant risk-free short rate), ## For 5%, enter 0.05
        sigma (volatility),                ## For 5%, enter 0.05
        d (continuous dividend rate),      ## For 5%, enter 0.05
        CP (call or put),                  ## Enter 'Call'/'C' or 'Put'/'P' in any case
        C (market price of the option)     ## Optional - only used for implied vol. method (default = 10)
        ------------------------------------------------------------------------------------------
        
        Methods:
        ------------------------------------------------------------------------------------------
        value (return present value of the option), 
        imp_vol (implied volatility given market price), 
        delta (option delta), 
        gamma (option gamma), 
        vega (option vega), 
        theta (option theta), 
        rho (option rho)
        ------------------------------------------------------------------------------------------
        '''
        
        self.S0 = S0
        self.K = K
        self.t = t
        self.M = M
        self.r = r
        self.sigma = sigma
        self.d = d
        self.CP = CP
        self.C = C
        self.refresh()
    
    def refresh(self):

        if type(self.t).__name__ == 'str':
            self.t= dt.strptime(self.t, '%m/%d/%Y')
        if type(self.M).__name__ == 'str':  
            self.M = dt.strptime(self.M, '%m/%d/%Y')
        if self.CP.lower() in ['call', 'c']:
            self.CP = 'call'
        elif self.CP.lower() in ['put', 'p']:
            self.CP = 'put'
        else:
            raise ValueError("Check value of variable CP - Call/C or Put/P allowed!")
        if self.t > self.M:
            raise ValueError("Pricing date later than maturity!")
        
        if type(self.t).__name__ in ['int', 'float']:
            self.T = self.M - self.t
        else:
            self.T = (self.M - self.t).days/365.0
		 
    
    def d1_d2(self):

        self.d1 = (log(self.S0 / self.K) + (self.r - self.d + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))
        self.d2 = self.d1 - self.sigma * sqrt(self.T)
        
    def value(self):
        
        self.refresh()
        self.d1_d2()
        
        if self.CP == 'call':
            value = (self.S0 * exp(-self.d * self.T) * stats.norm.cdf(self.d1, 0.0, 1.0) - self.K * exp(-self.r * self.T) * stats.norm.cdf(self.d2, 0.0, 1.0))
        else:
            value = (self.K * exp(-self.r * self.T) * stats.norm.cdf(-self.d2, 0.0, 1.0) - self.S0 * exp(-self.d * self.T) * stats.norm.cdf(-self.d1, 0.0, 1.0))
        return value
    
    def imp_vol(self):
        
        self.refresh()
        self.d1_d2()
        
        option = european_option(self.S0, self.K, self.t, self.M, self.r, self.sigma, self.d, self.CP, self.C)
        option.refresh()
        option.d1_d2()
        
        def difference(sig):
            option.sigma = sig
            return option.value() - option.C
        iv = fsolve(difference, option.sigma)[0]
        return iv
    
    def delta(self):
        
        self.refresh()
        self.d1_d2()
        
        if self.CP == 'call': 
            delta = exp(-self.d * self.T) * stats.norm.cdf(self.d1, 0.0, 1.0)
        else:
            delta = exp(-self.d * self.T) * (stats.norm.cdf(self.d1, 0.0, 1.0) - 1)    
        return delta
    
    def gamma(self):
        
        self.refresh()
        self.d1_d2()
        
        gamma = (exp(-self.d * self.T) * stats.norm.pdf(self.d1, 0.0, 1.0)) / (self.S0 * self.sigma * sqrt(self.T))
        return gamma
    
    def vega(self):
        
        self.refresh()
        self.d1_d2()
        
        vega = self.S0 * exp(-self.d * self.T) * stats.norm.pdf(self.d1, 0.0, 1.0) * sqrt(self.T)
        return vega
    
    def theta(self):
        
        self.refresh()
        self.d1_d2()
        
        if self.CP == 'call':
            theta = ( -(self.S0 * exp(-self.d * self.T) * stats.norm.pdf(self.d1, 0.0, 1.0) * self.sigma / (2 * sqrt(self.T)))
                        - (self.r * self.K * exp(-self.r * self.T) * stats.norm.cdf(self.d2, 0.0, 1.0))
                        + (self.d * self.S0 * exp(-self.d * self.T) * stats.norm.cdf(self.d1, 0.0, 1.0)))
        else:
            theta = ( -(self.S0 * exp(-self.d * self.T) * stats.norm.pdf(self.d1, 0.0, 1.0) * self.sigma / (2 * sqrt(self.T)))
                        + (self.r * self.K * exp(-self.r * self.T) * stats.norm.cdf(-self.d2, 0.0, 1.0))
                        - (self.d * self.S0 * exp(-self.d * self.T) * stats.norm.cdf(-self.d1, 0.0, 1.0)))
        return theta
    
    def rho(self):
        
        self.refresh()
        self.d1_d2()
        
        if self.CP == 'call':
            rho = self.K * self.T * exp(-self.r * self.T) * stats.norm.cdf(self.d2, 0.0, 1.0)
        else:
            rho = - self.K * self.T * exp(-self.r * self.T) * stats.norm.cdf(-self.d2, 0.0, 1.0)
        return rho
     


class RiskReversal:
    def __init__(self, S0, K1, K2, t, M, r, sigma, div):
        """
        Attributes:
        ------------------------------------------------------------------------------------------
        S0 (initial underlying level),     ## Risk reversal strategy: K1 < S0 < K2
        K1 (option strike price),          ## Selling an out-of-the-money put with strike K1
        K2 (option strike price),          ## Buying an out-of-the-money call with strike K2
        t (pricing date),                  ## Can enter in datetime format or str ('dd/mm/yyyy') or years (float)
        M (maturity date),                 ## Same as t
        r (constant risk-free short rate), ## For 5%, enter 0.05
        sigma (volatility),                ## For 5%, enter 0.05
        div (constant dividend rate),    ## For 5%, enter 0.05
        ------------------------------------------------------------------------------------------

        Methods:
        ------------------------------------------------------------------------------------------
        value (return present value of the risk reversal strategy),
        plot_payoff (plots net payoff diagram and present value for the range of underlying prices
                     [0.9 * K1, 1.1 * K2]).
        ------------------------------------------------------------------------------------------
        """

        self.S0 = S0
        self.K1 = K1
        self.K2 = K2
        self.t = t
        self.M = M
        self.r = r
        self.sigma = sigma
        self.div = div

        self.refresh()

    def refresh(self):
        if self.K1 < self.K2:
            pass
        else:
            raise ValueError("For risk reversal K1 < K2 is necessary!")

        if type(self.t).__name__ == 'str':
            self.t = dt.strptime(self.t, '%m/%d/%Y')
        if type(self.M).__name__ == 'str':
            self.M = dt.strptime(self.M, '%m/%d/%Y')

        if self.t > self.M:
            raise ValueError("Pricing date later than maturity!")

        if type(self.t).__name__ in ['int', 'float']:
            self.T = self.M - self.t
        else:
            self.T = (self.M - self.t).days / 365.0

    def d1_d2(self):
        ## d1 and d2 for put option
        self.d1_1 = (np.log(self.S0 / self.K1) + (self.r - self.div + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2_1 = self.d1_1 - self.sigma * np.sqrt(self.T)

        ## d1 and d2 for call option
        self.d1_2 = (np.log(self.S0 / self.K2) + (self.r - self.div + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2_2 = self.d1_2 - self.sigma * np.sqrt(self.T)

    def calculate_pv(self):
        self.refresh()
        self.d1_d2()

        value_put = (self.K1 * np.exp(-self.r * self.T) * stats.norm.cdf(-self.d2_1, 0.0, 1.0) - 
                     self.S0 * np.exp(-self.div * self.T) * stats.norm.cdf(-self.d1_1, 0.0, 1.0))
        value_call = (self.S0 * np.exp(-self.div * self.T) * stats.norm.cdf(self.d1_2, 0.0, 1.0) - 
                      self.K2 * np.exp(-self.r * self.T) * stats.norm.cdf(self.d2_2, 0.0, 1.0))

        pv_risk_reversal = value_call - value_put
        return pv_risk_reversal

    def plot_payoff(self):
        strategy_pv = []
        for i in range(int(0.8 * self.K1), int(1.1 * self.K2) + 1):
            options_pv = RiskReversal(i, self.K1, self.K2, self.t, self.M, self.r, self.sigma, self.div)
            self.po = max(i - self.K2, 0) - max(self.K1 - i, 0) - self.calculate_pv()
            strategy_pv.append([i, options_pv.calculate_pv(), self.po])

        np.set_printoptions(precision=3, suppress=True)
        self.strategy_payoff = np.array(strategy_pv)
        print(self.strategy_payoff)

        plt.plot(self.strategy_payoff[:, 0], self.strategy_payoff[:, 2])
        plt.xlabel('Underlying Prices')
        plt.ylabel('Net Payoff')
        plt.axhline(0, linewidth=0.5, color='grey')
        plt.show()

        plt.scatter(self.strategy_payoff[:, 0], self.strategy_payoff[:, 1], s=5, c='g')
        plt.xlabel('Present Value of Underlying Prices')
        plt.ylabel('Present Value of the Risk Reversal Strategy')
        plt.axhline(0, linewidth=0.5, color='grey')
        plt.show()
