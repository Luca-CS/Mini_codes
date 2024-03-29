
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

S = 100 #current price of the underlying security
K = 110 #strike price of the option
option = 1  #type of option: 1 for call, -1 for put
r = 0.03  #risk free rate
sigma = 0.25 #volatitly of underlying security
T = 1/12 #maturity of the option
Nsteps = 500 #number of time steps for the simulation
Nsim = 500 #number of simulated path for the price of the underlying security
Nopti = 500 #for control variate method, time periods to obtain optimal coefficient for reduced volatility

#Generates possible paths for the price of the underlying security using antithetic variate method

def paths_anti(option,S,r,sigma,T,Nsteps,Nsim):
    dt = T / Nsteps
    paths = np.zeros((Nsteps+1,Nsim))
    paths[0,:] = S
    
    for j in range(Nsim+1//2):
        for i in range(Nsteps):
            random = np.random.normal()
            paths[i+1,j] = paths[i,j] * np.exp(dt*(r-0.5*sigma**2)+sigma*np.sqrt(dt)*random)
            paths[i+1,-j] = paths[i,-j] * np.exp(dt*(r-0.5*sigma**2)-sigma*np.sqrt(dt)*random)
    payoffs = np.maximum(option*(paths[-1,:]-K), 0)
    mean = np.exp(-r*T)*np.mean(payoffs)
    return (paths,mean)

#Alternative generation of paths using control variate method (computing covariance between underlying and option price)

def paths_con(option,S,r,sigma,T,Nsteps,Nsim,Nopti):
    dt = T / Nsteps
    paths_1 = np.zeros((Nsteps+1,Nopti))
    paths_1[0,:] = S
    paths_2 = np.zeros((Nsteps+1,Nsim))
    paths_2[0,:] = S
    
    for j in range(Nopti):
        for i in range(Nsteps):
            random = np.random.normal()
            paths_1[i+1,j] = paths_1[i,j] * np.exp(dt*(r-0.5*sigma**2)+sigma*np.sqrt(dt)*random)
    payoffs_1 = np.maximum(option*(paths_1[-1,:]-K), 0)
    cov = np.cov(paths_1,payoffs_1)[1,0]
    var_z = S ** 2 * np.exp(2 * r * T) * (np.exp(T * sigma ** 2) - 1)
    c = -cov/var_z
    exp_z = S * np.exp(r*T)
    
    for j in range(Nsim):
        for i in range(Nsteps):
            random = np.random.normal()
            paths_2[i+1,j] = paths_2[i,j] * np.exp(dt*(r-0.5*sigma**2)+sigma*np.sqrt(dt)*random)
    
    payoffs_2 = np.maximum(option*(paths_2[-1,:]-K), 0) * np.exp(-r*T)
    mean = np.mean(payoffs_2 + c * (S - exp_z))
    return (paths_2, mean)

plt.figure()
plt.plot(paths_anti(option,S,r,sigma,T,Nsteps,Nsim)[0], color='green',linewidth=0.2)
plt.title("Antithetic variate method")
plt.xlabel("Time")
plt.ylabel("Value of the underlying security")
plt.figure()
plt.title("Control variate method")
plt.xlabel("Time")
plt.ylabel("Value of the underlying security")
plt.plot(paths_con(option,S,r,sigma,T,Nsteps,Nsim,Nopti)[0], color='red', linewidth=0.2)

#if does not look log normal, increase Nsim
plt.figure()    
plt.hist(paths_anti(option,S,r,sigma,T,Nsteps,Nsim)[0][:,-1])
plt.title("Distribution of underlying price at maturity (AVM)")
plt.figure()
plt.title("Distribution of underlying price at maturity (CVM)")
plt.hist(paths_con(option,S,r,sigma,T,Nsteps,Nsim,Nopti)[0][:,-1])


print("Prix anti:",paths_anti(option,S,r,sigma,T,Nsteps,Nsim)[1],"\n Prix con:", paths_con(option,S,r,sigma,T,Nsteps,Nsim,Nopti)[1])