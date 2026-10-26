from math import erf, exp, log, sqrt

import matplotlib.pyplot as plt
import numpy as np

S = 100  # current price of the underlying security
K = 110  # strike price of the option
option = 1  # type of option: 1 for call, -1 for put
r = 0.03  # risk free rate
sigma = 0.25  # volatility of underlying security
T = 1 / 12  # maturity of the option
Nsteps = 500  # number of time steps for the simulation
Nsim = 500  # number of simulated paths for the price of the underlying security
Nopti = 500  # for control variate method, pilot paths used to estimate the optimal coefficient


# Geometric Brownian motion paths from standard normal draws Z of shape (Nsteps, number of paths)

def gbm_paths(S, r, sigma, T, Nsteps, Z):
    dt = T / Nsteps
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.vstack([np.zeros(Z.shape[1]), np.cumsum(increments, axis=0)])
    return S * np.exp(log_paths)


def payoff(option, ST, K):
    return np.maximum(option * (ST - K), 0)


# Antithetic variate method: each path driven by Z is paired with the path driven by -Z

def paths_anti(option, S, K, r, sigma, T, Nsteps, Nsim, rng):
    half = Nsim // 2
    Z = rng.standard_normal((Nsteps, half))
    paths = np.hstack([gbm_paths(S, r, sigma, T, Nsteps, Z), gbm_paths(S, r, sigma, T, Nsteps, -Z)])
    pairs = 0.5 * (payoff(option, paths[-1, :half], K) + payoff(option, paths[-1, half:], K))
    price = np.exp(-r * T) * np.mean(pairs)
    stderr = np.exp(-r * T) * np.std(pairs, ddof=1) / np.sqrt(half)
    return paths, price, stderr


# Control variate method: the terminal price S_T is the control, with known mean S*e^(rT) and variance
# S^2*e^(2rT)*(e^(sigma^2*T) - 1). The optimal coefficient is estimated on independent pilot paths.

def paths_con(option, S, K, r, sigma, T, Nsteps, Nsim, Nopti, rng):
    pilot_ST = gbm_paths(S, r, sigma, T, Nsteps, rng.standard_normal((Nsteps, Nopti)))[-1]
    pilot_payoffs = np.exp(-r * T) * payoff(option, pilot_ST, K)
    cov = np.cov(pilot_payoffs, pilot_ST)[0, 1]
    var_z = S**2 * np.exp(2 * r * T) * (np.exp(T * sigma**2) - 1)
    c = -cov / var_z
    exp_z = S * np.exp(r * T)

    paths = gbm_paths(S, r, sigma, T, Nsteps, rng.standard_normal((Nsteps, Nsim)))
    ST = paths[-1]
    adjusted = np.exp(-r * T) * payoff(option, ST, K) + c * (ST - exp_z)
    return paths, np.mean(adjusted), np.std(adjusted, ddof=1) / np.sqrt(Nsim)


# Closed-form Black-Scholes price, used as the reference

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes(option, S, K, r, sigma, T):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return option * (S * norm_cdf(option * d1) - K * exp(-r * T) * norm_cdf(option * d2))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    anti, price_anti, se_anti = paths_anti(option, S, K, r, sigma, T, Nsteps, Nsim, rng)
    con, price_con, se_con = paths_con(option, S, K, r, sigma, T, Nsteps, Nsim, Nopti, rng)

    print(f"Black-Scholes            : {black_scholes(option, S, K, r, sigma, T):.4f}")
    print(f"Antithetic variate method: {price_anti:.4f} +/- {1.96 * se_anti:.4f} (95 %)")
    print(f"Control variate method   : {price_con:.4f} +/- {1.96 * se_con:.4f} (95 %)")

    plt.figure()
    plt.plot(anti, color="green", linewidth=0.2)
    plt.title("Antithetic variate method")
    plt.xlabel("Time")
    plt.ylabel("Value of the underlying security")
    plt.figure()
    plt.plot(con, color="red", linewidth=0.2)
    plt.title("Control variate method")
    plt.xlabel("Time")
    plt.ylabel("Value of the underlying security")

    # if does not look log normal, increase Nsim
    plt.figure()
    plt.hist(anti[-1])
    plt.title("Distribution of underlying price at maturity (AVM)")
    plt.figure()
    plt.hist(con[-1])
    plt.title("Distribution of underlying price at maturity (CVM)")
    plt.show()
