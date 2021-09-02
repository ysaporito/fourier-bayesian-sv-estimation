import numpy as np
'''
Simulation of various stochatic volatility processes
'''

np.random.seed(21) # fix seed for reproducibility

# in the functions below: T is the time horizon, n the number of simulted times (equally spaced), log(S0), v0 initial values for the process X, V and mu, kappa, m, xi, rho the parameters governing their dynamics

def HestonSimulation(S0, v0, mu, kappa, m, xi, rho, T, n):

    X = np.log(S0)*np.ones(n+1)
    V = v0*np.ones(n+1)

    dt = T/n

    dW_V = np.sqrt(dt)*np.random.normal(size=n)
    dW_V_perp = np.sqrt(dt)*np.random.normal(size=n)
    dW_X = rho*dW_V + np.sqrt(1.0 - rho**2)*dW_V_perp

    for i in range(n):
        V[i] = np.maximum(V[i], 0.0)
        X[i+1] = X[i] + (mu - 0.5*V[i])*dt + np.sqrt(V[i])*dW_X[i]
        V[i+1] = V[i] + kappa*(m - V[i])*dt + xi*np.sqrt(V[i])*dW_V[i]

    return X, V

def expOUSimulation(S0, v0, mu, kappa, m, xi, rho, T, n):

    X = np.log(S0)*np.ones(n+1)
    V = v0*np.ones(n+1)

    dt = T/n

    dW_V = np.sqrt(dt)*np.random.normal(size=n)
    dW_V_perp = np.sqrt(dt)*np.random.normal(size=n)
    dW_X = rho*dW_V + np.sqrt(1.0 - rho**2)*dW_V_perp

    for i in range(n):
        V_aux = np.exp(V[i])
        X[i+1] = X[i] + (mu - 0.5*V_aux**2)*dt + V_aux*dW_X[i]
        V[i+1] = V[i] + kappa*(m - V[i])*dt + xi*dW_V[i]

    return X, V

def GARCHSimulation(S0, v0, mu, kappa, m, xi, rho, T, n):

    X = np.log(S0)*np.ones(n+1)
    V = v0*np.ones(n+1)

    dt = T/n

    dW_V = np.sqrt(dt)*np.random.normal(size=n)
    dW_V_perp = np.sqrt(dt)*np.random.normal(size=n)
    dW_X = rho*dW_V + np.sqrt(1.0 - rho**2)*dW_V_perp

    for i in range(n):
        V[i] = np.maximum(V[i], 0.0)
        X[i+1] = X[i] + (mu - 0.5*V[i])*dt + np.sqrt(V[i])*dW_X[i]
        V[i+1] = V[i] + kappa*(m - V[i])*dt + xi*V[i]*dW_V[i]

    return X, V

def Simulation(S0, v0, param, T, n, model='Heston'):

    if model == 'Heston':
        X, V = HestonSimulation(S0, v0, param['mu'], param['kappa'], param['m'], 
                                param['xi'], param['rho'], T, n)

    elif model == 'Exp-OU':
        X, V = expOUSimulation(S0, v0, param['mu'], param['kappa'], param['m'], 
                                param['xi'], param['rho'], T, n)

    elif model == 'GARCH':
        X, V = GARCHSimulation(S0, v0, param['mu'], param['kappa'], param['m'],
                                param['xi'], param['rho'], T, n)

    else:
        print('Model not implemented')

    return X, V