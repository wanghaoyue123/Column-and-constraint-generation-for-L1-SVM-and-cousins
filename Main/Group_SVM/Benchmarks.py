
# from compare_group_Linf_SVM_CG      import *
import time
import os
#os.environ["GUROBI_HOME"]="/home/software/gurobi/gurobi811/linux64/"
#os.environ["GRB_LICENSE_FILE"]="/home/software/gurobi/gurobi.lic"

import numpy as np
import cvxpy as cp
import gurobipy



def use_Gurobi(X_train, y_train, lam, group_to_feat):
    
    N,P = np.shape(X_train)
    
    st = time.time()
    n_groups, _ = np.shape(group_to_feat)
    beta = cp.Variable((P,1))
    beta0 = cp.Variable()
    y_train_ = np.reshape(y_train,(N,1))
    loss = cp.sum(cp.pos(1 - cp.multiply(y_train_, X_train*beta + beta0)))
    
    reg = cp.sum([cp.norm(beta[group_to_feat[k]], 'inf') for k in range(n_groups)])
    prob = cp.Problem(cp.Minimize(loss + lam*reg))
    
    obj_gurobi = prob.solve(solver = 'GUROBI', verbose=True)
    
    ed = time.time()
    time_gurobi = ed - st
    
    
    return obj_gurobi, time_gurobi, np.reshape(beta.value, (P,)), beta0.value
    





def use_SCS(X_train, y_train, lam, group_to_feat):
    
    N,P = np.shape(X_train)
    
    st = time.time()
    n_groups, _ = np.shape(group_to_feat)
    beta = cp.Variable((P,1))
    beta0 = cp.Variable()
    y_train_ = np.reshape(y_train,(N,1))
    loss = cp.sum(cp.pos(1 - cp.multiply(y_train_, X_train*beta + beta0)))
    
    reg = cp.sum([cp.norm(beta[group_to_feat[k]], 'inf') for k in range(n_groups)])
    prob = cp.Problem(cp.Minimize(loss + lam*reg))
    
    obj_SCS = prob.solve(solver = 'SCS', verbose=True)
    
    ed = time.time()
    time_SCS = ed - st
    
    beta_ = np.reshape(beta.value, (P,))
    beta0_ = beta0.value
    constraints = np.ones(N) - y_train * (X_train.dot(beta_) + beta0_) 
    obj_SCS = np.sum(constraints*(constraints>0)) + lam* np.sum( [ np.linalg.norm(beta_[group_to_feat[k]], np.inf) for k in range(n_groups)] )
    
    return obj_SCS, time_SCS, beta_, beta0_
    
#     reg = cp.sum( [cp.norm(beta[0:10], 'inf')









