

import numpy as np
import datetime
import time
import os
import sys
import math
import subprocess

from gurobipy import *
from CVX_L0_Slope import *
from L0_Slope_CG import *
from simulate_data_classification import *
from smoothing_hinge_loss import *
from L1_SVM_CG import *
from proximal_slope_AGD import *


import signal
from scipy.sparse import *
import random

import os
# os.environ["GUROBI_HOME"]="/home/software/gurobi/gurobi811/linux64/"
# os.environ["GRB_LICENSE_FILE"]="/home/software/gurobi/gurobi.lic"






def use_FOM_CGCP(X_train, y_train, lam, compare_CVX, tol):
    
    N, P = np.shape(X_train)
    f = open('trash.txt', 'w')
    epsilon_RC  = 1e-2
    time_limit = 3600
    use_lagrangian = True
    st = time.time()
    
    alpha_list   = [lam]
    if not compare_CVX:
        K_Slope    = 0 
        lambda_arr = np.array([math.sqrt(math.log(2.*P/(j+1))) for j in range(P)])
    else:
        K_Slope    = 10
        lambda_arr = np.array([2 for _ in range(K_Slope)] + [1 for _ in range(P-K_Slope)])

    
    
    #---FIRST ORDER METHOD FOR SLOPE
    start_FOM       = time.time()
    P_reduced       = min(10*N, P)
    argsort_columns = np.argsort(np.abs(np.dot(X_train.T, y_train) ))
    index_CG        = argsort_columns[::-1][:P_reduced]
    X_train_reduced = np.array([X_train[:,j] for j in index_CG]).T 

    tau_max = 0.2
    n_loop  = 1
    n_iter  = 200

    #     write_and_print('\n\n###### FIRST ORDER METHOD K_SLOPE=P #####', f)
    # Parameters
    if not compare_CVX:
        X_add                        = 1./math.sqrt(N)*np.ones((N,P_reduced+1))
        X_add[:,:P_reduced]          = X_train_reduced
        mu_max                       = power_method(X_add)
        time_slope, beta_FOM, beta_0 = loop_proximal_slope_AGD('hinge', X_train_reduced, y_train, alpha_list[0]*lambda_arr, np.zeros(P_reduced+1), X_add, f, highest_eig=mu_max)
        #index_columns_FOM = np.where(beta_FOM!=0)[0]
        #beta_FOM          = beta_FOM[index_columns_FOM]

    else:
        _, index_columns_FOM, time_slope, beta_FOM, _ = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced, y_train, alpha_list[0], tau_max, n_loop, n_iter, f)

    index_columns_FOM = np.where(beta_FOM!=0)[0]
    beta_FOM          = beta_FOM[index_columns_FOM]
    index_columns_FOM = np.array(index_CG)[index_columns_FOM].tolist()	

    index_columns_FOM_K_Slope = np.argsort(np.abs(beta_FOM))[::-1]#[-K_Slope:]
    w_star_FOM_Slope          = np.zeros(len(index_columns_FOM))	
    for j in range(len(index_columns_FOM)): w_star_FOM_Slope[index_columns_FOM_K_Slope[j]] = lambda_arr[j]
    
    delta = 0
    ed1 = time.time()
    
    model_method_1 = 0
    [beta, beta0], support_, time_, model_, index_columns_, obj_ = L0_Slope_CG(X_train, y_train, 
                                                                       w_star_FOM_Slope, index_columns_FOM, 
                                                                       alpha_list[0], K_Slope, 
                                                                       lambda_arr, use_lagrangian, 
                                                                       delta, epsilon_RC, 
                                                                       time_limit, model_method_1, 
                                                                       [], f)
    ed2 = time.time()
    time_total = ed2 - st
    time_CGCP = ed2 - ed1
    
    return obj_, time_total, time_CGCP, beta, beta0









def use_FOM(X_train, y_train, lam, lambda_arr):
    
    N,P = np.shape(X_train)
    f = open('trash.txt', 'w')

    ###################  ###################
    #### ONLY RUN ONE ITERATION
    class TimeoutException(Exception):   # Custom exception class
        pass
    def timeout_handler(signum, frame):   # Custom signal handler
        raise TimeoutException

    write_and_print('\n###### FOM to high accuracy #####', f)

    
    #### STOP AFTER 3h
    signal.signal(signal.SIGALRM, timeout_handler)
    TIME_LIMIT = 10800
    signal.alarm(TIME_LIMIT)    

    try:
        start_FOM   = time.time()
        X_add       = 1./math.sqrt(N)*np.ones((N,P+1))
        X_add[:,:P] = X_train
        mu_max      = power_method(X_add)
        n_iter      = 1e6
        n_loop      = 1

        time_FOM_slope, beta_FOM, beta_0 = loop_proximal_slope_AGD('hinge', X_train, y_train, lam*lambda_arr, np.zeros(P+1), X_add, f, highest_eig=mu_max, n_iter=n_iter, n_loop=n_loop)
        objval_FOM = compute_objval_slope(X_train, y_train, beta_FOM, beta_0, lam*lambda_arr, f)
    except:
        write_and_print('\nTIME all regularization path CG = '+str(time_method_1_tot), f)

    signal.alarm(0)

    
    
    return objval_FOM, time_FOM_slope











