import numpy as np
import time
import math

import sys

from soft_thresholding_slope import *



def proximal_slope_AGD(type_loss, X, y, alpha_arr, beta_start, X_add, f, highest_eig=0, tau=.2, n_iter=200):
    
#TYPE_LOSS = 'HINGE'  : smoothing hinge loss, TAU controls the smoothing
#TYPE_LOSS = 'LOG_REG': logistic regression

#X_ADD:     matrix of input with a column of 1
#ALPHA_ARR: array of penalizations
    
    
#---Initialisation
    start_time = time.time()
    N, P  = X.shape

    old_beta = -np.ones(P+1)
    beta     = beta_start


#---Lipschitz constant
    dict_Lipschit_coeff = {'hinge': highest_eig/(4*tau), 'logreg': 0.25*np.linalg.norm(X, ord='fro')**2 }
    Lipchtiz_coeff      = dict_Lipschit_coeff[type_loss]


#---MAIN LOOP   
    t_AGD_old = 1
    t_AGD     = 1
    eta_old   = beta_start
    ones_N    = np.ones(N)    

    test  =0
    while np.linalg.norm(beta - old_beta)>1e-5 and test < n_iter: 
        #print np.linalg.norm(beta-old_beta)
        test+=1
        
    #---Define gradient
        if (type_loss=='hinge'):
            aux           = ones_N - y*np.dot(X_add, beta)
            w_tau         = [min(1, abs(aux[i])/(2*tau))*np.sign(aux[i])  for i in range(N)]
            gradient_aux  = np.array([y[i]*(1+w_tau[i]) for i in range(N)])
            gradient_loss = -0.5*np.dot(X_add.T, gradient_aux)


        if (type_loss=='logreg'):
            aux           = ones_N + np.exp(y*np.dot(X_add, beta))
            gradient_aux  = y/aux
            gradient_loss = -np.dot(X_add.T, gradient_aux)


    #---Gradient descent
        old_beta = beta 
        grad     = beta - 1/float(Lipchtiz_coeff)*gradient_loss

    #---Slope thresholding
        eta = np.array(soft_thresholding_slope(grad[:P], alpha_arr/Lipchtiz_coeff).tolist() + [grad[P]])

    #---AGD
        t_AGD     = (1 + math.sqrt(1+4*t_AGD_old**2))/2.
        aux_t_AGD = (t_AGD_old-1)/t_AGD

        beta       = eta + aux_t_AGD * (eta - eta_old)

        t_AGD_old = t_AGD
        eta_old   = eta
  

    write_and_print('Number of iterations: ' +str(test), f)
    write_and_print('Convergence: ' +str(round(np.linalg.norm(beta-old_beta) ,4)), f)

    beta[P] /= math.sqrt(N)

#---Support
    support_slope = np.where(beta[:P] !=0)[0]
    write_and_print('Len support proximal: '+str(support_slope.shape[0]), f)
    #print('Support: '+str(support_slope), f)

#---Time
    time_slope = time.time()-start_time
    write_and_print('Time proximal: '+str(round(time_slope,3)), f)


    return time_slope, beta






def loop_proximal_slope_AGD(type_loss, X, y, alpha_arr, beta_start, X_add, f, highest_eig=0, tau_max=.2, n_loop=1, n_iter=200):
    
#n_loop : how many times should we run the loop ?
#n_iter : number of iterations per loop
    

    #write_and_print('\n\n\n#### SLOPE FOR ALPHA= '+str(alpha_arr[-1]/math.log(2)), f)
    start_time = time.time()
    N, P       = X.shape
    old_beta   = -np.ones(P+1)

    test = 0
    tau = tau_max

    #while(np.linalg.norm(beta_start-old_beta)>1e-3 and test < n_loop): 
    while test < n_loop:
        #print '\n## Iter '+str(test)+ ' CV: '+str(round(np.linalg.norm(beta_start-old_beta), 4))
        test     += 1
        old_beta  = beta_start
        
        time_slope, beta_start = proximal_slope_AGD(type_loss, X, y, alpha_arr, beta_start, X_add, f, highest_eig=highest_eig, tau=tau, n_iter=n_iter)

        if False:
        #if test == 1:
        #---Restrict columns
            support       = np.where(beta_start[:P]!=0)[0]
            X             = X[:, support]
            X_add         = X_add[:, list(support)+[P]]
            highest_eig   = power_method(X_add)
            beta_start    = beta_start[list(support)+[P]]

    #---Update parameters        
        tau    = 0.5*tau



    time_tot = time.time()-start_time
    write_and_print('\n## Number of iterations outer loop: '+str(test), f)
    write_and_print('## Total time smoothing for '+str(type_loss)+': '+str(round(time_tot, 3)), f)

    support       = np.where(beta_start[:P]!=0)[0]
    beta, beta_0 = beta_start[:-1], beta_start[-1]
    beta_tot     = np.zeros(P)
    for i in range(len(support)): beta_tot[support[i]] = beta[i]

    return time_tot, beta, beta_0



    
def write_and_print(text,f):
    print(text)
    f.write('\n'+text)





#POWER METHOD to compute the SVD of XTX
def power_method(X):
    P = X.shape[1]

    highest_eigvctr     = np.random.rand(P)
    old_highest_eigvctr = -1
    
    while(np.linalg.norm(highest_eigvctr - old_highest_eigvctr)>1e-2):
        old_highest_eigvctr = highest_eigvctr
        highest_eigvctr     = np.dot(X.T, np.dot(X, highest_eigvctr))
        highest_eigvctr    /= np.linalg.norm(highest_eigvctr)
    
    X_highest_eig = np.dot(X, highest_eigvctr)
    highest_eig   = np.dot(X_highest_eig.T, X_highest_eig)/np.linalg.norm(highest_eigvctr)
    return highest_eig






def compute_objval_slope(X_train, y_train, beta_FOM, beta_0, lambda_arr, f):
    N,P    = X_train.shape
    xi     = np.ones(N) - y_train*(np.dot(X_train, beta_FOM) + beta_0*np.ones(N))
    objval = np.sum([max(xi[i], 0) for i in range(N)])

    idx_argsort = np.argsort(np.abs(beta_FOM))[::-1]
    for j in range(P):
        objval += lambda_arr[j]*abs(beta_FOM[idx_argsort[j]])

    write_and_print('Obj value FOM   = '+str(objval), f)
    return objval








