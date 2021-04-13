import numpy as np
import time
import math

import sys

# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


sys.path.append('../../SPGL1_python_port')
# from spgl1 import spg_lasso
from spgl1.spgl1 import spg_lasso



def smoothing_proximal_group_Linf(type_loss, type_penalization, X, y, group_to_feat, alpha, beta_start, X_add, highest_eig, tau, n_iter, f):
    
#TYPE_PENALIZATION = 1 : L1 or L2 -> soft thresholding
#TYPE_PENALIZATION = 2 : L1_INF   -> soft thresholding with l1 projection
#TYPE_PENALIZATION = 2 : L1_INF_CD-> soft thresholding with l1 projection and block coordinate descent

#BETA_M : the last component is beta_0 the origin coefficient
    
    
#---Initialisation
    start_time = time.time()
    N, P  = X.shape
    G     = len(group_to_feat) #number of groups

    old_beta = -np.ones(P+1)
    
    beta_m = beta_start

    

#---MAIN LOOP   
    test  =0
    t_AGD_old =1
    t_AGD     =1
    eta_m_old = beta_start
    
    # New
    ones_N       = np.ones(N)
    X_beta_group = np.dot(X_add, beta_start)
    active_loop  = np.array([True for _ in range(G)])
    
    len_supp_act_loop     = 0
    old_len_supp_act_loop = -1


    if (type_loss=='hinge'):
        Lipchtiz_coeff  = highest_eig/(4*tau) 
    elif (type_loss=='squared_hinge'):
        Lipchtiz_coeff = 2*highest_eig

    

    while(np.linalg.norm(beta_m-old_beta)>1e-3 and len_supp_act_loop != old_len_supp_act_loop and test < n_iter):
        test += 1
        old_beta = np.copy(beta_m) 


    ### CASE OF BLOCK COORIDNATE DESCENT
        for idx in range(Lipchtiz_coeff.shape[0]):

            for _ in range(1):


            ### Is the group in the active set of groups ?
                if active_loop[idx] == True:

                #---Update 1-y*XT*beta
                    if type_penalization == 'l1_linf_CD':
                        aux = ones_N - y*X_beta_group
                    else:    
                        aux = ones_N - y*np.dot(X_add, beta_m)


                ### Compute gradient
                    if (type_loss=='hinge'):
                        w_tau         = [min(1, abs(aux[i])/(2*tau))*np.sign(aux[i])  for i in range(N)]
                        gradient_aux  = np.array([y[i]*(1+w_tau[i]) for i in range(N)])

                    #---If block CD then only need gradient on group
                        if type_penalization == 'l1_linf_CD':
                            gradient_loss = -0.5*np.dot(X_add[:, list(group_to_feat[idx])+[P]].T, gradient_aux)
                        else:
                            gradient_loss = -0.5*np.dot(X_add.T, gradient_aux)


                    if (type_loss=='squared_hinge'):
                        xi            = np.array([max(0, aux[i]) for i in range(N)])
                        gradient_loss = -2*np.dot(X_add.T, y*xi)



                ### Gradient step + Thresholding operators
                    if type_penalization == 'l1_linf_CD':
                        old_beta_m_group  = np.copy(beta_m[list(group_to_feat[idx])+[P]])
                        grad              = beta_m[list(group_to_feat[idx])+[P]] - 1./Lipchtiz_coeff[idx] * gradient_loss
                        soft_thresholding = soft_thresholding_linf(grad[:-1], alpha/Lipchtiz_coeff[idx]) 

                    #---Check if the group has not been set to zero
                        block_not_null = not np.all([soft_thresholding_coef == 0 for soft_thresholding_coef in soft_thresholding])

                        active_loop[idx]           = block_not_null
                        beta_m[group_to_feat[idx]] = soft_thresholding
                        beta_m[P]                  = grad[-1]


                    #---Residual updates
                        delta_beta_group   = np.dot(X_add[:, list(group_to_feat[idx])+[P]], beta_m[list(group_to_feat[idx])+[P]]-old_beta_m_group)
                        X_beta_group      += delta_beta_group
                    

                    else:
                        grad = beta_m - 1/float(np.array(Lipchtiz_coeff)[idx])*gradient_loss   #Not block CD
                        
                        dict_thresholding = {'l1':         soft_thresholding_l1,
                                             'l2':         soft_thresholding_l2,
                                             'l1_linf':    soft_thresholding_l1_linf
                                             }
                        eta_m = np.array(dict_thresholding[type_penalization](grad[:P], alpha/Lipchtiz_coeff[idx], group_to_feat) + [grad[P]]) #idx = 0
                    
                    #---AGD (not for block CD ??)
                        t_AGD     = (1 + math.sqrt(1+4*t_AGD_old**2))/2.
                        aux_t_AGD = (t_AGD_old-1)/t_AGD

                        beta_m     = eta_m + aux_t_AGD * (eta_m - eta_m_old)

                        t_AGD_old = t_AGD
                        eta_m_old = eta_m

                        active_loop = np.array([not np.all([beta_m[i] == 0 for i in group_to_feat[j]]) for j in range(G)])


    #---Test whether same sparsity
        old_len_supp_act_loop = min(len_supp_act_loop, G/2+.5) #dont want to stop during first iterations
        len_supp_act_loop     = len(np.where(active_loop==True)[0])
        print(len_supp_act_loop)



    print('Test CV')
    print(np.linalg.norm(beta_m-old_beta))

    write_and_print('\nNumber of iterations: ' +str(test), f)

    b0 = beta_m[P]/math.sqrt(N)


#---Support
    idx_groups_smoothing = np.where(active_loop==True)[0]

    write_and_print('Len group support smoothing: '+str(len(idx_groups_smoothing)), f)
    write_and_print('Convergence rate           : ' +str(round(np.linalg.norm(beta_m-old_beta), 3)), f) 
    time_smoothing = time.time()-start_time
    write_and_print('Time smoothing: '+str(round(time_smoothing,3)), f)

    return idx_groups_smoothing, time_smoothing, beta_m



    


def loop_smoothing_proximal_group_Linf(type_loss, type_penalization, X, y, group_to_feat, alpha, tau_max, n_loop, time_limit, n_iter, f):
    
#n_loop: how many times should we run the loop ?
#Apply the smoothing technique from the best subset selection
    
    start_time = time.time()
    N, P = X.shape
    old_beta = -np.ones(P+1)


#---Lipschitz coeff
    X_add       = 1/math.sqrt(N)*np.ones((N, P+1))
    X_add[:,:P] = X

    if type_penalization == 'l1_linf_CD':
        n_groups    = len(group_to_feat)
        highest_eig = np.zeros(n_groups)

        for i in range(n_groups):
            len_group               = len(group_to_feat[i])
            X_add_bis               = 1/math.sqrt(N)*np.ones((N, len_group+1))
            X_add_bis[:,:len_group] = X[:,group_to_feat[i]]
            highest_eig[i]          = power_method(X_add)

        #Run block CD for decreasing sequence of alpha
        alpha_list = [5*alpha, 2*alpha, alpha]

    else:
        highest_eig = np.array([power_method(X_add)])
        alpha_list  = [alpha]


    beta_smoothing  = np.zeros(P+1)
    time_smoothing_sum = 0


    for alpha in alpha_list:
        tau = tau_max
        test= 0

        write_and_print('\nAlpha : '+str(alpha), f)
        while(np.linalg.norm(beta_smoothing-old_beta)>1e-4 and test < n_loop): 

            test += 1
            old_beta = np.copy(beta_smoothing)
            
            idx_groups, time_smoothing, beta_smoothing = smoothing_proximal_group_Linf(type_loss, type_penalization, X, y, group_to_feat, alpha, beta_smoothing, X_add, highest_eig, tau, n_iter, f)

        #---Update parameters
            time_smoothing_sum += time_smoothing
            tau = 0.7*tau


    time_smoothing_tot = time.time()-start_time
    write_and_print('\nNumber of iterations                       : '+str(test), f)
    write_and_print('Total time smoothing for '+str(type_loss)+': '+str(round(time_smoothing_tot, 3)), f)

    return idx_groups, time_smoothing_sum, beta_smoothing










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



def soft_thresholding_l1(c,alpha,group_to_feat):
    result = np.copy(c)

    for i in range(c.shape[0]):

        if(alpha>=abs(c[i])):
            result[i] = 0
        else:
            if (c[i]>=0):
                result[i] = c[i]-alpha
            else:
                result[i] = c[i]+alpha

    return list(result)
    
    
def soft_thresholding_l2(c,alpha,group_to_feat):
    return c/float(1+2*alpha)


def soft_thresholding_linf(c, alpha):
    if np.sum(np.abs(c)) > alpha:
        proj_l1_ball, _, _, _  = spg_lasso(np.identity(c.shape[0]), c, alpha)
    else:
        proj_l1_ball = c
    return c - proj_l1_ball


def soft_thresholding_l1_linf(c,alpha,group_to_feat):
    result = np.zeros(c.shape[0])

    for i in range(len(group_to_feat)):
        c_group       = c[group_to_feat[i]]
        l1_norm_group = np.sum(np.abs(c_group))
        
        if l1_norm_group > alpha:
            proj_l1_ball_group, _, _, _       = spg_lasso(np.identity(c_group.shape[0]), c_group, alpha)
            result[group_to_feat[i]]          = c_group - proj_l1_ball_group
    
    return list(result)



def soft_thresholding_l1_linf_CB(c, alpha, group_to_feat):
    result = np.copy(c)

    c_group       = c[group_to_feat]
    l1_norm_group = np.sum(np.abs(c_group))
    
    if l1_norm_group > alpha:
        proj_l1_ball_group, _, _, _    = spg_lasso(np.identity(c_group.shape[0]), c_group, alpha)
    else:
        proj_l1_ball_group = c_group
        
    result[group_to_feat] = c_group - proj_l1_ball_group
    return list(result)




