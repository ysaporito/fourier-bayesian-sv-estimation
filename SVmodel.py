'''
Simulation of various stochatic volatility processes
'''
import numpy as np

np.random.seed(21) # fix seed for reproducibility

def HestonSimulation(S, v, mu, kappa, m, xi, rho, T, n):

    X = np.log(S)*np.ones(n+1)
    V = v*np.ones(n+1)

    dt = T/n

    dW_V = np.sqrt(dt)*np.random.normal(size=n)
    dW_V_perp = np.sqrt(dt)*np.random.normal(size=n)
    dW_X = rho*dW_V + np.sqrt(1.0 - rho**2)*dW_V_perp

    for i in range(n):
        V[i] = np.maximum(V[i], 0.0)
        X[i+1] = X[i] + (mu - 0.5*V[i])*dt + np.sqrt(V[i])*dW_X[i]
        V[i+1] = V[i] + kappa*(m - V[i])*dt + xi*np.sqrt(V[i])*dW_V[i]

    return X, V

def expOUSimulation(S, v, mu, kappa, m, xi, rho, T, n):

    X = np.log(S)*np.ones(n+1)
    V = v*np.ones(n+1)

    dt = T/n

    dW_V = np.sqrt(dt)*np.random.normal(size=n)
    dW_V_perp = np.sqrt(dt)*np.random.normal(size=n)
    dW_X = rho*dW_V + np.sqrt(1.0 - rho**2)*dW_V_perp

    for i in range(n):
        V_aux = np.exp(V[i])
        X[i+1] = X[i] + (mu - 0.5*V_aux**2)*dt + V_aux*dW_X[i]
        V[i+1] = V[i] + kappa*(m - V[i])*dt + xi*dW_V[i]

    return X, V

def GARCHSimulation(S, v, mu, kappa, m, xi, rho, T, n):

    X = np.log(S)*np.ones(n+1)
    V = v*np.ones(n+1)

    dt = T/n

    dW_V = np.sqrt(dt)*np.random.normal(size=n)
    dW_V_perp = np.sqrt(dt)*np.random.normal(size=n)
    dW_X = rho*dW_V + np.sqrt(1.0 - rho**2)*dW_V_perp

    for i in range(n):
        V[i] = np.maximum(V[i], 0.0)
        X[i+1] = X[i] + (mu - 0.5*V[i]**2)*dt + np.sqrt(V[i])*dW_X[i]
        V[i+1] = V[i] + kappa*(m - V[i])*dt + xi*V[i]*dW_V[i]

    return X, V

def Simulation(S, v, param, T, n, model='Heston'):

    if model == 'Heston':
        X, V = HestonSimulation(S, v, param['mu'], param['kappa'], param['m'], 
                                param['xi'], param['rho'], T, n)

    elif model == 'Exp-OU':
        X, V = expOUSimulation(S, v, param['mu'], param['kappa'], param['m'], 
                                param['xi'], param['rho'], T, n)

    elif model == 'GARCH':
        X, V = GARCHSimulation(S, v, param['mu'], param['kappa'], param['m'],
                                param['xi'], param['rho'], T, n)

    else:
        print('Model not implemented')

    return X, V