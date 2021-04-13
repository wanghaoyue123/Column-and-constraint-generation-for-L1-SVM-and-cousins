import numpy as np
import time
import math
import sys


from simulate_data_classification import *
from scipy.sparse import *





# STARTING POINT FOR L1-SVM

def smoothing_hinge_loss(type_loss, type_penalization, X, y, alpha, beta_start, X_add, highest_eig, tau, n_iter, f, is_sparse=False):
    
#TYPE_PENALIZATION = 1 : L1 -> soft thresholding
#TYPE_PENALIZATION = 2 : L2
#BETA_M : the last component is beta_0 the origin coefficient
    
    
#---Initialization
    start_time = time.time()
    N, P  = X.shape

    old_beta = np.ones(P+1)
    beta_m   = beta_start

    


#---MAIN LOOP   
    test  =0
    t_AGD_old =1
    t_AGD     =1
    eta_m_old = beta_start
    ones_N    = np.ones(N)


    if (type_loss=='hinge'):
        Lipchtiz_coeff  = highest_eig/(4*tau) 
    elif (type_loss=='squared_hinge'):
        Lipchtiz_coeff = 2*highest_eig
    


    while(np.linalg.norm(beta_m-old_beta)>1e-3 and test < n_iter): 
        test+=1
        aux = ones_N - y*np.dot(X_add,beta_m) if not is_sparse else ones_N - y*X_add.dot(beta_m)
        

    #---Hinge loss
        if (type_loss=='hinge'):
            w_tau           = [min(1, abs(aux[i])/(2*tau))*np.sign(aux[i])  for i in range(N)]
            #gradient_loss   = -0.5*np.sum([y[i]*(1+w_tau[i])*X_add[i,:] for i in range(N)], axis=0)
            gradient_aux  = np.array([y[i]*(1+w_tau[i]) for i in range(N)])
            
            gradient_loss = -0.5*np.dot(X_add.T, gradient_aux) if not is_sparse else -0.5*X_add.T.dot(gradient_aux)




    #---Gradient descent
        old_beta = beta_m 
        grad     = beta_m - 1/float(Lipchtiz_coeff)*gradient_loss


    #---Thresholding of top 100 guys !
        dict_thresholding = {'l1': soft_thresholding_l1,
                             'l2': soft_thresholding_l2}
        eta_m = np.array([ dict_thresholding[type_penalization](grad[i], alpha/Lipchtiz_coeff) for i in range(P)] + [grad[P]])
   

    #---AGD
        t_AGD     = (1 + math.sqrt(1+4*t_AGD_old**2))/2.
        aux_t_AGD = (t_AGD_old-1)/t_AGD

        beta_m     = eta_m + aux_t_AGD * (eta_m - eta_m_old)

        t_AGD_old = t_AGD
        eta_m_old = eta_m

    
    

    write_and_print('\nNumber of iterations: ' +str(test), f)
    write_and_print('Shape: ' +str(X.shape), f)

#---Keep top 50
    #index    = np.abs(beta_m[:P]).argsort()[::-1][:50]
    #index = range(P)
    b0 = beta_m[P]/math.sqrt(N)
    #write_and_print('intercept: '+str(b0), f) #very small


#---Support
    idx_columns_smoothing   = np.where(beta_m[:P] !=0)[0]
    write_and_print('Len support smoothing: '+str(idx_columns_smoothing.shape[0]), f)


#---Constraints
    ##### USE B0 !!!!!!!
    if not is_sparse:
        constraints = 1.05*ones_N - y*( np.dot(X[:,idx_columns_smoothing], beta_m[idx_columns_smoothing]) + b0* ones_N) 
    else:
        constraints = 1.05*ones_N - y*( X[:,idx_columns_smoothing].dot( beta_m[idx_columns_smoothing]) + b0* ones_N) 

    idx_samples_smoothing = np.arange(N)[constraints >= 0]
    write_and_print('Number violated constraints: '+str(idx_samples_smoothing.shape[0]), f)
    write_and_print('Convergence rate    : ' +str(round(np.linalg.norm(beta_m-old_beta), 3)), f) 
    
    time_smoothing = time.time()-start_time
    write_and_print('Time smoothing: '+str(round(time_smoothing,3)), f)


    return idx_samples_smoothing.tolist(), idx_columns_smoothing.tolist(), time_smoothing, beta_m



    








def loop_smoothing_hinge_loss(type_loss, type_penalization, X, y, alpha, tau_max, n_loop, n_iter, f, is_sparse=False):
    
#n_loop: how many times should we run the loop ?
#Apply the smoothing technique from the best subset selection
    
    start_time = time.time()
    N, P = X.shape
    old_beta = -np.ones(P+1)

#---New matrix and SVD
    if not is_sparse:
        X_add       = 1/math.sqrt(N)*np.ones((N, P+1))
        X_add[:,:P] = X
        highest_eig = power_method(X_add)
    else: 
        X_add = csr_matrix( hstack([X, coo_matrix(1/math.sqrt(N)*np.ones((N,1)))]) )
        highest_eig = power_method(X_add, is_sparse=True)



    beta_smoothing  = np.zeros(P+1)
    time_smoothing_sum = 0

    tau = tau_max
    
    test = 0
    while(np.linalg.norm(beta_smoothing-old_beta)>1e-3 and test < n_loop): 
        print('TEST CV BEFORE TAU: '+str(np.linalg.norm(beta_smoothing-old_beta)))

        test += 1
        old_beta = beta_smoothing
        
        idx_samples, idx_columns, time_smoothing, beta_smoothing = smoothing_hinge_loss(type_loss, type_penalization, X, y, alpha, beta_smoothing, X_add, highest_eig, tau, n_iter, f, is_sparse)

    #---Update parameters
        time_smoothing_sum += time_smoothing
        tau = 0.7*tau


    #print beta_smoothing[idx_columns]

    time_smoothing_tot = time.time()-start_time
    write_and_print('\nNumber of iterations              : '+str(test), f)
    write_and_print('Total time smoothing for '+str(type_loss)+': '+str(round(time_smoothing_tot, 3)), f)

    return idx_samples, idx_columns, time_smoothing_sum, beta_smoothing[:-1], beta_smoothing[-1]







def loop_smoothing_hinge_loss_samples_restricted(type_loss, type_penalization, X, y, alpha, tau_max, n_loop, n_iter, f):
    
#n_loop: how many times should we run the loop ?
#Apply the smoothing technique from the best subset selection
    
    start_time = time.time()
    N, P = X.shape
    old_beta = -np.ones(P+1)


#---New matrix and SVD
    X_add       = 1/math.sqrt(N)*np.ones((N, P+1))
    X_add[:,:P] = X
    highest_eig = power_method(X_add)


#---Results
    beta_smoothing  = np.zeros(P+1)
    time_smoothing_sum = 0
    tau = tau_max


#---Prepare for restrcition
    idx_samples = np.arange(N)
    X_reduced   = X
    y_reduced   = y
    
    
    test = -1
    while(np.linalg.norm(beta_smoothing-old_beta)>1e-4 and test < n_loop): 
        print('TEST CV BEFORE TAU')
        print(np.linalg.norm(beta_smoothing-old_beta))

        test += 1
        old_beta = beta_smoothing
        
        idx_samples_restricted, _, time_smoothing, beta_smoothing = smoothing_hinge_loss(type_loss, type_penalization, X_reduced, y_reduced, alpha, beta_smoothing, X_add, highest_eig, tau, n_iter, f)


        if test == 0:
        #---Restrict to samples
            X_reduced = X_reduced[idx_samples_restricted,:] 
            y_reduced = np.array(y)[idx_samples_restricted]
            N_reduced = len(idx_samples_restricted)

            X_add         = 1/math.sqrt(N)*np.ones((N_reduced, P+1))
            X_add[:,:P]   = X_reduced
            highest_eig   = power_method(X_add)
            idx_samples   = idx_samples[idx_samples_restricted]
    

    #---Update parameters        
        time_smoothing_sum += time_smoothing
        tau = 0.7*tau


    time_smoothing_tot = time.time()-start_time
    write_and_print('\nNumber of iterations              : '+str(test), f)
    write_and_print('Total time smoothing for '+str(type_loss)+': '+str(round(time_smoothing_tot, 3)), f)

    return idx_samples.tolist(), time_smoothing_sum, beta_smoothing







def loop_smoothing_hinge_loss_columns_samples_restricted(type_loss, type_penalization, X, y, alpha, tau_max, n_loop, n_iter, f, is_sparse=False, beta_init=None):
    
#n_loop: how many times should we run the loop ?
#Apply the smoothing technique from the best subset selection
    
    start_time = time.time()
    N, P = X.shape
    old_beta   = -np.ones(P+1)


#---New matrix and SVD
    if not is_sparse:
        X_add       = 1/math.sqrt(N)*np.ones((N, P+1))
        X_add[:,:P] = X
        highest_eig = power_method(X_add)


    else: 
        X_add = csr_matrix( hstack([X, coo_matrix(1/math.sqrt(N)*np.ones((N,1)))]) )
        highest_eig = power_method(X_add, is_sparse=True)



#---Results
    beta_smoothing  = np.zeros(P+1)
    if beta_init is not None:
        beta_smoothing[:P] = beta_init

    time_smoothing_sum = 0
    tau = tau_max


#---Prepare for restriction
    X_reduced    = X
    y_reduced    = y
    idx_columns_restricted = np.arange(P)
    
    
    test = -1
    while(np.linalg.norm(beta_smoothing-old_beta)>1e-2 and test < n_loop): 
        print('TEST CV BETWEEN 2 VALUES OF TAU: '+str(np.linalg.norm(beta_smoothing-old_beta)))
        if test == 0:
            old_beta  = np.concatenate([beta_smoothing[idx_columns_restricted], [beta_smoothing[-1]] ])
        else:
            old_beta  = beta_smoothing
        
        test += 1
        idx_samples_restricted, idx_columns_restricted, time_smoothing, beta_smoothing = smoothing_hinge_loss(type_loss, type_penalization, X_reduced, y_reduced, alpha, old_beta, X_add, highest_eig, tau, n_iter, f, is_sparse)

        if test == 0:
        #---Dont change samples -> just restrict columns
            X_reduced = X_reduced[:, idx_columns_restricted]
            P_reduced = X_reduced.shape[1]

            X_add         = X_add[:, idx_columns_restricted+[P]]
            highest_eig   = power_method(X_add, is_sparse)
            idx_columns   = idx_columns_restricted
    



    #---Update parameters        
        time_smoothing_sum += time_smoothing
        tau    = 0.7*tau
        n_iter = 50

#---Results
    beta_smoothing_sample = np.zeros(P+1)
    for i in range(len(idx_columns)): beta_smoothing_sample[idx_columns[i]] = beta_smoothing[i]

    time_smoothing_tot = time.time()-start_time
    write_and_print('\nNumber of iterations              : '+str(test), f)
    write_and_print('Total time smoothing for '+str(type_loss)+': '+str(round(time_smoothing_tot, 3)), f)

    #return idx_samples.tolist(), idx_columns.tolist(), time_smoothing_sum, beta_smoothing
    return beta_smoothing_sample
















#POWER METHOD to compute the SVD of XTX

def power_method(X, is_sparse=False):
    P = X.shape[1]

    highest_eigvctr     = np.random.rand(P)
    old_highest_eigvctr = -1
    
    while(np.linalg.norm(highest_eigvctr - old_highest_eigvctr)>1e-2):
        old_highest_eigvctr = highest_eigvctr
        highest_eigvctr     = np.dot(X.T, np.dot(X, highest_eigvctr)) if not is_sparse else X.T.dot(X.dot(highest_eigvctr))
        highest_eigvctr    /= np.linalg.norm(highest_eigvctr)
    
    X_highest_eig = np.dot(X, highest_eigvctr) if not is_sparse else X.dot(highest_eigvctr)

    highest_eig   = np.dot(X_highest_eig.T, X_highest_eig)/np.linalg.norm(highest_eigvctr)
    return highest_eig



def soft_thresholding_l1(c,alpha):

    if(alpha>=abs(c)):
        return 0
    else:
        if (c>=0):
            return c-alpha
        else:
            return c+alpha
    
    
def soft_thresholding_l2(c,alpha):
    return c/float(1+2*alpha)




