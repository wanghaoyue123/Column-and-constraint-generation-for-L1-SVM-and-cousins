

import os
#os.environ["GUROBI_HOME"]="/home/software/gurobi/gurobi811/linux64/"
#os.environ["GRB_LICENSE_FILE"]="/home/software/gurobi/gurobi.lic"

import numpy as np
import gurobipy

import datetime
import time
import os
import sys
import math
import subprocess


from gurobipy import *
from group_Linf_SVM_CG import *
# from group_Linf_SVM_CG_plots import *
from simulate_data_group import *
from smoothing_proximal_group_Linf import *


from smoothing_hinge_loss import *





def use_FO1_CG(X_train, y_train, lam, group_to_feat, tol=0.01):
    N,P = np.shape(X_train)
    f = open('trash.txt', 'w')
    num_groups, _ = np.shape(group_to_feat)
    
    st = time.time()
    ## corr screening 
    abs_correlations  = np.abs(np.dot(X_train.T, y_train) )
    sum_correl_groups = [np.sum(abs_correlations[idx]) for idx in group_to_feat]
    N_groups          = 10
    index_groups      = np.argsort(sum_correl_groups)[::-1][:N_groups]

    X_train_reduced       = np.zeros((N,0))
    group_to_feat_reduced = []
    aux = 0

    # Order in groups
    for i in range(N_groups):
        X_train_reduced = np.concatenate([X_train_reduced, np.array([X_train[:,j] for j in group_to_feat[index_groups[i]]]).T ], axis=1)
        group_to_feat_reduced.append(range(aux, aux+len(group_to_feat[index_groups[i]]) ))
        aux += len(group_to_feat[index_groups[i]]) 
        
        
    ###### Nesterov method without CD #####
    tau_max = .2
    n_loop  = 1
    n_iter  = 50
    time_limit = 3600
    index_groups_FO1, time_smoothing_FO1, beta_FO1 = loop_smoothing_proximal_group_Linf('hinge', 'l1_linf', 
                                                                                   X_train_reduced, y_train, 
                                                                                   group_to_feat_reduced, lam, 
                                                                                   tau_max, n_loop, 
                                                                                   time_limit, n_iter, f)
    index_groups_FO1 = np.array(index_groups)[index_groups_FO1].tolist()
    
    ed1 = time.time()
    
    ###### GROUP_LINF SVM WITH AGD + CG, WITHOUT BLOCK CD #####
    model_FO1_CG = 0
    [beta_, beta0], support_, time_, model_, index_columns_, obj_ = group_Linf_SVM_CG(X_train, y_train, 
                                                                              group_to_feat, index_groups_FO1, 
                                                                              lam, tol, 
                                                                              time_limit, model_FO1_CG, [], f)

    ed2 = time.time()
    
    time_total = ed2 - st
    time_CG = ed2 - ed1
    
    beta_full = np.zeros(P)
    i = 0
    for k in index_columns_:
        beta_full[group_to_feat[k]] = beta_[10*i:10*(i+1)]
        i+=1
        
#     constraints = np.ones(N) - y_train * (X_train.dot(beta_full) + beta0) 
#     obj2 = np.sum(constraints*(constraints>0)) + lam* np.sum( [ np.linalg.norm(beta_full[group_to_feat[k]], np.inf) for k in range(num_groups)] )
    
#     print("sss=", obj2 - obj_)

        
    return obj_, time_total, time_CG, beta_full, beta0
    

    
    

    
    




def use_FO2_CG(X_train, y_train, lam, group_to_feat, tol=0.01):
    N,P = np.shape(X_train)
    f = open('trash.txt', 'w')
    num_groups, _ = np.shape(group_to_feat)
    
    st = time.time()
    ## corr screening 
    abs_correlations  = np.abs(np.dot(X_train.T, y_train) )
    sum_correl_groups = [np.sum(abs_correlations[idx]) for idx in group_to_feat]
    N_groups          = 10
    index_groups      = np.argsort(sum_correl_groups)[::-1][:N_groups]

    X_train_reduced       = np.zeros((N,0))
    group_to_feat_reduced = []
    aux = 0

    # Order in groups
    for i in range(N_groups):
        X_train_reduced = np.concatenate([X_train_reduced, np.array([X_train[:,j] for j in group_to_feat[index_groups[i]]]).T ], axis=1)
        group_to_feat_reduced.append(range(aux, aux+len(group_to_feat[index_groups[i]]) ))
        aux += len(group_to_feat[index_groups[i]]) 
        
        
    ###### Nesterov method with CD #####
    tau_max = .2
    n_loop  = 1
    n_iter  = 50
    time_limit = 3600
    index_groups_FO2, time_smoothing_FO2, beta_FO2 = loop_smoothing_proximal_group_Linf('hinge', 'l1_linf_CD', 
                                                                                   X_train_reduced, y_train, 
                                                                                   group_to_feat_reduced, lam, 
                                                                                   tau_max, n_loop, 
                                                                                   time_limit, n_iter, f)
    index_groups_FO2 = np.array(index_groups)[index_groups_FO2].tolist()
    
    ed1 = time.time()
    
    ###### GROUP_LINF SVM WITH AGD + CG, WITH BLOCK CD #####
    model_FO2_CG = 0
    [beta_, beta0], support_, time_, model_, index_columns_, obj_ = group_Linf_SVM_CG(X_train, y_train, 
                                                                              group_to_feat, index_groups_FO2, 
                                                                              lam, tol, 
                                                                              time_limit, model_FO2_CG, [], f)

    ed2 = time.time()
    
    time_total = ed2 - st
    time_CG = ed2 - ed1
    
    beta_full = np.zeros(P)
    i = 0
    for k in index_columns_:
        beta_full[group_to_feat[k]] = beta_[10*i:10*(i+1)]
        i+=1
    
#     constraints = np.ones(N) - y_train * (X_train.dot(beta_full) + beta0) 
#     obj2 = np.sum(constraints*(constraints>0)) + lam* np.sum( [ np.linalg.norm(beta_full[group_to_feat[k]], np.inf) for k in range(num_groups)] )
    
#     print("sss=", obj2 - obj_)
    

        
    return obj_, time_total, time_CG, beta_full, beta0
    








def use_RP_CG(X_train, y_train, lam, alpha_max, group_to_feat, tol=0.01):
    N,P = np.shape(X_train)
    f = open('trash.txt', 'w')
    num_groups, _ = np.shape(group_to_feat)
    print("num_groups=", num_groups)
    
    st = time.time()
    ## corr screening 
    abs_correlations  = np.abs(np.dot(X_train.T, y_train) )
    sum_correl_groups = [np.sum(abs_correlations[idx]) for idx in group_to_feat]
    N_groups          = 10
    index_groups      = np.argsort(sum_correl_groups)[::-1][:N_groups]

    X_train_reduced       = np.zeros((N,0))
    group_to_feat_reduced = []
    aux = 0

    # Order in groups
    for i in range(N_groups):
        X_train_reduced = np.concatenate([X_train_reduced, np.array([X_train[:,j] for j in group_to_feat[index_groups[i]]]).T ], axis=1)
        group_to_feat_reduced.append(range(aux, aux+len(group_to_feat[index_groups[i]]) ))
        aux += len(group_to_feat[index_groups[i]]) 
        
        
    ###### REGULARIZATION PATH GROUP LINF SVM #####
    alpha_bis = alpha_max
    beta_RP_CG     = []
    time_RP_CG_tot = 0
    n_groups = 5 ####CHECK (50 before)
    idx_groups_CG, time_correl = init_group_Linf(X_train, y_train, group_to_feat, n_groups, f)
    model_RP_CG = 0
    time_limit = 3600

    while 0.7*alpha_bis > lam:
        beta_RP_CG, support_RP_CG, time_RP_CG, model_RP_CG, idx_groups_CG, obj_RP_CG   = group_Linf_SVM_CG(X_train, y_train, 
                                                                                                           group_to_feat, idx_groups_CG, 
                                                                                                           alpha_bis, tol, 
                                                                                                           time_limit, model_RP_CG, 
                                                                                                           beta_RP_CG, f)
        alpha_bis   *= 0.7
        time_RP_CG_tot += time_RP_CG
    [beta_, beta0], support_RP_CG, time_RP_CG, model_RP_CG, idx_groups_CG, obj_RP_CG   = group_Linf_SVM_CG(X_train, y_train, 
                                                                                                       group_to_feat, idx_groups_CG, 
                                                                                                       lam, tol, 
                                                                                                       time_limit, model_RP_CG, 
                                                                                                       beta_RP_CG, f)

    ed = time.time()
    
    time_total = ed - st

    
    beta_full = np.zeros(P)
    i = 0
    for k in idx_groups_CG:
        beta_full[group_to_feat[k]] = beta_[10*i:10*(i+1)]
        i+=1
        
#     constraints = np.ones(N) - y_train * (X_train.dot(beta_full) + beta0) 
#     obj2 = np.sum(constraints*(constraints>0)) + lam* np.sum( [ np.linalg.norm(beta_full[group_to_feat[k]], np.inf) for k in range(num_groups)] )
    
#     print("sss=", obj2 - obj_RP_CG)
    
    return obj_RP_CG, time_total, beta_full, beta0
    







