import numpy as np
import math
from aux_algorithm1_classification import *


def algorithm1_classification(type_loss, type_penalization, X, y, K0, alpha, beta_start, X_add, epsilon, highest_eig=0, tau=0.2, n_iter=500):
    
#TYPE LOSS        : one of the parameters squared_hinge', 'hinge', 'logreg'
#TYPE PENALIZATION: one of the parameters l1', 'l2'

#XTY, mu_max : computed once
#START       : warm start of the form (beta, beta_0) (or ([],0))
#EPSILON     : convergence criterion

#TAU         : used to smooth non differentiable hinge loss

    N,P = X.shape

#---Beta
    old_beta = -np.ones(P+1)
    beta     = np.zeros(P+1) if len(beta_start) == 0 else np.concatenate( [beta_start[0], np.array([beta_start[1]]) ] ) #form of start


#---Lipschitz constant
    #*np.linalg.norm(X_add, ord='fro')**2
    dict_Lipschit_coeff = {'squared_hinge': 2., 'hinge': 0.25/tau, 'logreg': 0.25}
    Lipchtiz_coeff      = dict_Lipschit_coeff[type_loss]*highest_eig
    
    
#---Main loop
    test   = 0
    ones_N = np.ones(N)

    while np.linalg.norm(beta-old_beta) > epsilon and test < n_iter: 
        test += 1
        aux  = y*np.dot(X_add, beta)

        old_beta = np.copy(beta)

    #---Squared hinge loss
        if type_loss == 'squared_hinge':
            gradient_aux  = np.array([y[i]*max(0, 1-aux[i]) for i in range(N)])
            gradient_loss = - 2*np.dot(X_add.T, gradient_aux)

    #---Hinge loss
        if type_loss == 'hinge':
            w_tau         = [min(1, abs(1 - aux[i])/(2.*tau))*np.sign(1 - aux[i])  for i in range(N)]
            gradient_aux  = np.array([y[i]*(1+w_tau[i]) for i in range(N)])
            gradient_loss = - 0.5*np.dot(X_add.T, gradient_aux)

    #---LogReg loss
        elif (type_loss=='logreg'):
            gradient_aux  = np.array([y[i]/(1 + math.exp( min(max(1e-2, aux[i]), 1e2)) ) for i in range(N)]) #math range error
            gradient_loss = - np.dot(X_add.T, gradient_aux)
        

    #---Gradient descent
        grad = beta - 1/float(Lipchtiz_coeff)*gradient_loss
        for i in range(P): beta[i] = 0

    #---Soft-thresholding
        dict_thresholding = {'l1':   soft_thresholding_l1,
                             'l2':   soft_thresholding_l2}

        index            = np.abs(grad[:P]).argsort()[::-1][:K0]
        grad_thresholded = np.array([ dict_thresholding[type_penalization](grad[idx], alpha/Lipchtiz_coeff) for idx in index])
        beta[index]      = grad_thresholded
        beta[P]          = grad[P]  #intercept term


#---Run estimator on support 
    #print test, np.linalg.norm(beta-old_beta)
    beta, beta_0, error = estimator_on_support(type_loss, type_penalization, X, y, alpha, beta)

    return beta, beta_0, error
    






def loop_tau_algorithm1_with_hinge(type_penalization, X, y, K0, alpha, beta_start, X_add, epsilon, highest_eig=0, tau_max=0.2, n_loop=20, n_iter=500):
    
#N_LOOP     : how many times should we run the loop
#BETA_START : of the form (beta, beta0)
    
    N, P = X.shape
    tau  = tau_max
    
#---Not intercept in CV
    old_beta       = -np.ones(P)
    beta_smoothing = np.zeros(P) if beta_start[0].shape[0] == 0 else beta_start[0][:P]
    beta_0         = beta_start[1]

    test = 0
    while np.linalg.norm(beta_smoothing - old_beta) > 1e-2 and test < n_loop: 
        test    += 1
        old_beta = beta_smoothing
        
        beta_smoothing, beta_0, error = algorithm1_classification('hinge', type_penalization, X, y, K0, alpha, (beta_smoothing, beta_0), X_add, epsilon, highest_eig=highest_eig, tau=tau, n_iter=n_iter)
        tau = 0.7*tau

    return beta_smoothing, beta_0, error





def algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start, X_add, epsilon, highest_eig=0, n_iter=500):

    if type_loss == 'hinge':
        return loop_tau_algorithm1_with_hinge(type_penalization, X, y, K0, alpha, beta_start, X_add, epsilon, highest_eig=highest_eig, n_iter=n_iter)
    elif type_loss == 'squared_hinge' or type_loss == 'logreg':
        return algorithm1_classification(type_loss, type_penalization, X, y, K0, alpha, beta_start, X_add, epsilon, highest_eig=highest_eig, n_iter=n_iter)



