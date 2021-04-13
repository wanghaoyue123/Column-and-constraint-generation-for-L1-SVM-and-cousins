import numpy as np
from gurobipy import *

import sys
# sys.path.append('../L1_SVM_CG')
from L1_SVM_CG import *

# sys.path.append('../algorithms')
from smoothing_hinge_loss import *

from sklearn.svm import LinearSVC
import time

from scipy.sparse import *
import random




def init_both_CG_CP_sampling_smoothing(X_train, y_train, alpha, rho, f, is_sparse=False, max_n_cols=None, beta_init=[]):

    N, P = X_train.shape
        
    #N0   = int(max(np.sqrt([N,P])))
    #N0   = int(max(N/10, P/10))
    N0, P0 = 5*int(math.sqrt(N)), 5*int(math.sqrt(P))

    start_time = time.time()


    #RESULT
    old_beta_averaged  = -np.ones(P+1)
    beta_averaged      = np.zeros(P+1)
    delta_l2_diff_mean = 1e6

    k = 0
    while delta_l2_diff_mean>5e-1 and k<int(N/N0):
        k += 1
        write_and_print('\n\n\n###### SAMPLE NUMBER '+str(k), f)
        write_and_print('###### DIFFERENCE L2 NORMS '+str(delta_l2_diff_mean), f)

        subset = np.sort(random.sample(range(N),N0))
        X_train_reduced = X_train[subset, :] 
        y_train_reduced = y_train[subset] 



    #---Correlation threshold
        if not is_sparse:
            #argsort_columns = np.argsort(np.abs(np.dot(X_train.T, y_train) ))
            argsort_columns = np.argsort(np.abs(np.dot(X_train_reduced.T, y_train_reduced) ))
        else:
            argsort_columns = np.argsort(np.abs( X_train_reduced.T.dot(y_train_reduced) ))
            X_train_reduced = X_train_reduced.A


        #N_columns       = 10*N0
        index_CG        = argsort_columns[::-1][:P0]
        X_train_reduced = np.array([X_train_reduced[:,j] for j in index_CG]).T

        if len(beta_init) > 0:
            assert len(beta_init) == P
            beta_init_reduced = np.array([beta_init[j] for j in index_CG]).T
        else:
            beta_init_reduced = None

    ## Haoyue: changed here!!!
#     #---Normalize
#         l2_X_train_reduced = []
#         for i in range(len(index_CG)):
#             l2 = np.linalg.norm(X_train_reduced[:,i])         
#             l2_X_train_reduced.append(l2)
#             X_train_reduced[:,i] /= l2
#         #write_and_print('Reduced matrix shape: '+str(X_train_reduced.shape), f)
        
    ## Haoyue: changed here!!!
        #Classical alpha
#         alpha_sample = 1e-2*np.max(np.sum( np.abs(X_train_reduced), axis=0)) 
        alpha_sample = rho * np.max(np.sum( np.abs(X_train_reduced), axis=0)) 



        if is_sparse: X_train_reduced = csr_matrix(X_train_reduced)


    #---First order method
        tau_max = 0.2
        n_loop  = 20
        n_iter  = 100
        beta_sample_reduced = loop_smoothing_hinge_loss_columns_samples_restricted('hinge', 'l1', X_train_reduced, y_train_reduced, alpha_sample, tau_max, n_loop, n_iter, f, is_sparse, beta_init=beta_init_reduced)


        beta_sample = np.zeros(P+1)
        
        ## Haoyue: changed here!!!
#         for i in range(len(index_CG)): beta_sample[index_CG[i]] = beta_sample_reduced[i]/l2_X_train_reduced[i]
        for i in range(len(index_CG)): beta_sample[index_CG[i]] = beta_sample_reduced[i]


        old_beta_averaged = np.copy(beta_averaged)
        beta_averaged    += np.array(beta_sample)

        delta_l2_diff_mean = np.linalg.norm(1./max(1,k)*beta_averaged - 1./max(1,k-1)*old_beta_averaged)


#---Average
    beta_averaged *= 1./k
    b0_averaged   = beta_averaged[-1]
    beta_averaged = beta_averaged[:-1]
    b0_averaged = 0
    ones_N = np.ones(N)


#---Determine set of columns
    idx_columns_smoothing = np.where(beta_averaged != 0)[0]
    print('Len support primal: '+str(len(idx_columns_smoothing)))

    if max_n_cols is None:
        max_n_cols = 200
    idx_columns_smoothing = np.argsort(np.abs(beta_averaged))[::-1][:max_n_cols]


#---Determine set of constraints
    constraints = 1.01*ones_N - y_train*( np.dot(X_train, beta_averaged) + b0_averaged*ones_N) if not is_sparse else 1.01*ones_N - y_train*( X_train.dot(beta_averaged) + b0_averaged*ones_N)
    idx_samples_smoothing = np.arange(N)[constraints >= 0]
    write_and_print('Len dual smoothing: '+str(idx_samples_smoothing.shape[0]), f)

    time_smoothing = time.time()-start_time
    write_and_print('Total time: '+str(round(time_smoothing,3)), f)
    return list(idx_samples_smoothing), list(idx_columns_smoothing), time_smoothing




