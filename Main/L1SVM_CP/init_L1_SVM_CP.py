import numpy as np
from gurobipy import *

import sys
# sys.path.append('../L1_SVM_CG')
from L1_SVM_CG import *

# sys.path.append('../algorithms')
from smoothing_hinge_loss import *

from sklearn.svm import LinearSVC
import time

import random




def init_CP_sampling_smoothing(X_train, y_train, alpha, rho, is_restricted, f):

    N, P = X_train.shape
    N0   = int(min(10*P, N/4)) ##not 2 samples for N=2k

    start_time = time.time()

    tau_max = 0.1
    n_loop  = 20
    n_iter  = 20


    #RESULT
    old_beta_averaged = -np.ones(P+1)
    beta_averaged     = np.zeros(P+1)
    delta_variance    = 1e6

    k = 0
    while delta_variance>5e-2 and k< int(N/N0):
        k += 1
        write_and_print('\n\n\n###### SAMPLE NUMBER '+str(k), f)
        write_and_print('###### DIFFERENCE VARIANCE '+str(delta_variance), f)

        subset = np.sort(random.sample(range(N),N0))
        X_train_reduced = X_train[subset, :] 
        y_train_reduced = y_train[subset] 
    
    ## Haoyue: changed here!!
#     #---Normalize
#         l2_X_train_reduced = []
#         for i in range(P):
#             l2 = np.linalg.norm(X_train_reduced[:,i])         
#             l2_X_train_reduced.append(l2)
#             X_train_reduced[:,i] = X_train_reduced[:,i]/l2

    ## Haoyue: changed here!!
#         alpha_sample_0 = alpha*N0/N
#         alpha_sample = 0.01*np.max(np.sum( np.abs(X_train_reduced), axis=0))  #roughly alpha*np.sqrt(N0/float(N))
        alpha_sample = rho*np.max(np.sum( np.abs(X_train_reduced), axis=0))

        if is_restricted:
            n_loop  = 10
            _, _, beta_sample       = loop_smoothing_hinge_loss_samples_restricted('hinge', 'l1', X_train_reduced, y_train_reduced, alpha_sample, tau_max, n_loop, n_iter, f)
        else:
            _, _, _, beta_sample, beta0_sample = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced, y_train_reduced, alpha_sample, tau_max, n_loop, n_iter, f)
            beta_sample = np.concatenate([ beta_sample, np.array([beta0_sample]) ])

    ## Haoyue: changed here!
#         for i in range(P): beta_sample[i] /= l2_X_train_reduced[i]

        old_beta_averaged = np.copy(beta_averaged)
        beta_averaged    += np.array(beta_sample)
        delta_variance    = np.linalg.norm(1./max(1,k)*beta_averaged - 1./max(1,k-1)*old_beta_averaged)



#---Determine set of constraints
    beta_averaged *= N0/float(N)
    b0_averaged   = beta_averaged[-1]
    beta_averaged = beta_averaged[:-1]
    b0_averaged = 0
    ones_N = np.ones(N)


    constraints = 1*ones_N - y_train*( np.dot(X_train, beta_averaged) + b0_averaged*ones_N)
    idx_samples_smoothing = np.arange(N)[constraints >= 0]
    write_and_print('\n\n\nFINISHED', f)
    write_and_print('Len dual smoothing: '+str(idx_samples_smoothing.shape[0]), f)

    time_smoothing = time.time()-start_time
    write_and_print('Total time: '+str(round(time_smoothing,3)), f)
    return list(idx_samples_smoothing), time_smoothing





def liblinear_for_CP(type_liblinear, X, y, alpha, f):

    write_and_print('\n\n\n###### USE SCIKIT #####', f)
    start  = time.time()
    N      = X.shape[0]
    ones_N = np.ones(N)

    if type_liblinear== 'hinge_l2':
        estimator = LinearSVC(penalty='l2', loss= 'hinge', dual=True, C=1/(2*float(alpha)))
    elif type_liblinear== 'squared_hinge_l1':
        estimator = LinearSVC(penalty='l1', loss= 'squared_hinge', dual=False, C=1/float(alpha))

    estimator = estimator.fit(X, y)

    beta_liblinear, b0 = estimator.coef_[0], estimator.intercept_[0]
    constraints        = 1*ones_N - y*( np.dot(X, beta_liblinear) + b0*ones_N )

    idx_liblinear = np.arange(N)[constraints >= 0].tolist()
    write_and_print('Len dual liblinear: '+str(len(idx_liblinear)), f)

    time_liblinear = time.time()-start
    write_and_print('Time liblinear for '+type_liblinear+': '+str(time_liblinear), f)

    return list(idx_liblinear), time_liblinear, beta_liblinear

























########################## OLDER APPROACHES ##########################



########## This approach takes the closest sample to an hyperplan ########## 

def init_CP_clustering(X_train, y_train, n_samples, f):

    start = time.time()

    X_plus = X_train[y_train==1]
    mean_plus = np.mean(X_plus, axis=0)

    X_minus = X_train[y_train==-1]
    mean_minus = np.mean(X_minus, axis=0)


#---Hyperplan between two centroids
    vect_orth_hyperplan = mean_plus - mean_minus
    point_hyperplan     = 0.5*(mean_plus + mean_minus)
    b0 = -np.dot(point_hyperplan, vect_orth_hyperplan)


#---Rank point according to distance to one mean or to distance to hyperplan .??

#--- +1 class
    dist_X_plus   = [np.dot(X_plus[i,:] - point_hyperplan, vect_orth_hyperplan) + b0 for i in range(X_plus.shape[0])]
    index_CP_plus = np.array(np.abs(dist_X_plus)).argsort()[:n_samples/2]
    X_init_plus   = np.matrix([X_plus[i,:] for i in index_CP_plus])


#--- -1 class
    dist_X_minus    = [np.dot(X_minus[i,:] - point_hyperplan, vect_orth_hyperplan) + b0 for i in range(X_minus.shape[0])]
    index_CP_minus  = np.array(np.abs(dist_X_minus)).argsort()[:n_samples/2]
    X_init_minus    = np.matrix([X_minus[i,:] for i in index_CP_minus])
    

    index_CP = np.concatenate([index_CP_plus, index_CP_minus])
    write_and_print('Time init: '+str(time.time()-start), f)

    return index_CP





########## This approach tries to generalize the ideas of ranking the columns with absolute correlations ########## 

def init_CP_norm_samples(X_train, y_train, n_samples, f):
    start = time.time()
    sum_lines = np.sum(np.abs(X_train), axis=1)
    
    argsort_lines = np.argsort(sum_lines)
    index_CP      = argsort_lines[::-1][:n_samples].tolist()
    
    time_l1_norm = time.time()-start
    write_and_print('Time l1 norm:'+str(time_l1_norm), f)
    
    return index_CP, time_l1_norm





########## This approach removes hald of the constraints at every iteration by solving an RFE kind of method on the top 10 features ########## 

def init_CP_dual(X, y, alpha, n_samples, f):
    N, P = X.shape

    index_columns = init_correlation(X, y, 10, f)

    X_RFE  = np.array([ [X[i][j] for j in index_columns] for i in range(N)])
    idx_CP = restrict_lines_CP_dual(X_RFE, y, alpha, n_samples, f)

    return idx_CP



def restrict_lines_CP_dual(X, y, alpha, n_samples, f):
    start = time.time()
    N, P = X.shape
    iteration_limit = 1

    idx_CP       = range(N)
    n_constrains = N
    dual_L1_SVM_CP = dual_L1_SVM_CP_model(X, y, idx_CP, alpha, iteration_limit)

    while n_constrains > n_samples:

    #---Optimize model
        dual_L1_SVM_CP.optimize()
        pi = np.array([dual_L1_SVM_CP.getVarByName("pi_"+str(i)) for i in idx_CP])
        

    #---Rank constraints
        n_constraints_to_remove = min(n_constrains/2, n_constrains - n_samples)
        remove_constraints      = np.array(pi).argsort()[:n_constraints_to_remove] #pi>0
        idx_to_removes          = np.array(idx_CP)[remove_constraints]
        n_constrains           -= n_constraints_to_remove


    #---Remove constraints
        pis_to_remove = np.array([dual_L1_SVM_CP.getVarByName(name="pi_"+str(i)) for i in idx_to_removes])

        for pi_to_remove in pis_to_remove:
            dual_L1_SVM_CP.remove(pi_to_remove)
        dual_L1_SVM_CP.update()
            
        for remove_constraint in idx_to_removes:
            idx_CP.remove(remove_constraint)


    write_and_print('Time heuristic for sample subset selection: '+str(time.time()-start)+'\n', f)

    return idx_CP



def dual_L1_SVM_CP_model(X, y, idx_CP, alpha, iteration_limit):

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P  = X.shape
    N_CP = len(idx_CP)

#---VARIABLES
    dual_L1_SVM_CP=Model("dual_L1_SVM_CP")
    dual_L1_SVM_CP.setParam('OutputFlag', False )
    #dual_L1_SVM_CP.setParam('IterationLimit', iteration_limit)
    
    
    #Hinge loss
    pi = np.array([dual_L1_SVM_CP.addVar(lb=0, name="pi_"+str(idx_CP[i])) for i in range(N_CP)])
    dual_L1_SVM_CP.update()


#---OBJECTIVE VALUE 
    dual_L1_SVM_CP.setObjective(quicksum(pi), GRB.MINIMIZE)


#---PI CONSTRAINTS 
    for i in range(N_CP):
        dual_L1_SVM_CP.addConstr(pi[i] <= 1, name="pi_"+str(i))


#---PI CONSTRAINTS 
    for j in range(P):
        dual_L1_SVM_CP.addConstr(quicksum([ y[idx_CP[i]] * X[idx_CP[i]][j]*pi[i] for i in range(N_CP)]) <= alpha,  name="dual_beta_+_"+str(idx_CP[i]))
        dual_L1_SVM_CP.addConstr(quicksum([ y[idx_CP[i]] * X[idx_CP[i]][j]*pi[i] for i in range(N_CP)]) >= -alpha, name="dual_beta_-_"+str(idx_CP[i]))



#---ORTHOGONALITY
    dual_L1_SVM_CP.addConstr(quicksum([ pi[i]*y[idx_CP[i]] for i in range(N_CP)]) == 0, name='orthogonality')

  
#---RESULT
    dual_L1_SVM_CP.update()
    return dual_L1_SVM_CP













