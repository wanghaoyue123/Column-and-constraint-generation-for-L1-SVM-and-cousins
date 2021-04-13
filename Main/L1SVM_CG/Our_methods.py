import os
# os.environ["GUROBI_HOME"]="/home/software/gurobi/gurobi811/linux64/"
# os.environ["GRB_LICENSE_FILE"]="/home/software/gurobi/gurobi.lic"

from smoothing_hinge_loss import *
import time
from gurobipy import *
from L1_SVM_CG import *



def use_FOM_CG(X_train, y_train, lam, tau_max, tol=1e-2):
    #---FIRST ORDER METHOD
    N, P = np.shape(X_train)
    f = open('trash.txt', 'w')
    st       = time.time()
    argsort_columns = np.argsort(np.abs( X_train.T.dot(y_train) ))
    index_CG        = argsort_columns[::-1][:10*N]
    X_train_reduced = X_train[:, index_CG]
    
    n_loop  = 1
    n_iter  = 200
    
    idx_sample, idx_col, tm, beta, beta0= loop_smoothing_hinge_loss('hinge', 
                                                                    'l1', 
                                                                    X_train_reduced, 
                                                                    y_train, 
                                                                    lam, 
                                                                    tau_max, 
                                                                    n_loop, 
                                                                    n_iter, 
                                                                    f, 
                                                                    is_sparse=False)
    idx_col = np.array(index_CG)[idx_col].tolist()
    print("len(idx_col)=", len(idx_col))
    
    time_limit = 3600
    _model = 0
    ed1 = time.time()
    
    [_beta,beta0], support, _time, _model, idx_col, obj = L1_SVM_CG(X_train, 
                                                           y_train, 
                                                           idx_col, 
                                                           lam, 
                                                           tol, 
                                                           time_limit, 
                                                           _model, 
                                                           [], 
                                                           f, 
                                                           is_sparse=False, 
                                                           dict_nnz={})
    
    ed2 = time.time()
    time_total = ed2 - st
    time_CG = ed2 - ed1
    
    beta = np.zeros(P)
    beta[support] = _beta
    
    
    return obj, time_total, time_CG, beta, beta0
    
    
    

    
    
    
def use_RP_CG(X_train, y_train, lam, lam_max, tol=1e-2):
    
    st = time.time()
    f = open('trash.txt', 'w')
    N, P = np.shape(X_train)
    lam_bis = lam_max
    idx_col, _  = init_correlation(X_train, y_train, 10, f)
    beta = []
    _model    = 0
    time_limit = 3600
    obj = 0
    
    while 0.5*lam_bis > lam:
        beta, support, _time, _model, idx_col, obj   = L1_SVM_CG(X_train, y_train, 
                                                                 idx_col, lam_bis, 
                                                                 tol, time_limit, 
                                                                 _model, beta, f)
        lam_bis   *= 0.5
    
    [_beta, beta0], support, _time, _model, idx_col, obj = L1_SVM_CG(X_train, y_train, 
                                                           idx_col, lam, 
                                                           tol, time_limit, 
                                                           _model, beta, f)
        
    beta_RP_CG = np.zeros(P)
    beta_RP_CG[support] = _beta
    ed = time.time()
    time_RP_CG = ed - st
    obj_RP_CG = obj
    beta0_RP_CG = beta0
    
    return obj_RP_CG, time_RP_CG, beta_RP_CG, beta0_RP_CG    
    
    
    
    
    
    
    
    
def use_random_init_CG(X_train, y_train, lam, n_features, tol=1e-2):
    
    N, P = np.shape(X_train)
    f = open('trash.txt', 'w')
    st       = time.time()
    idx_col = random.sample(range(P), n_features)
    _model = 0
    time_limit = 10000
    [_beta,beta0], support, _time, _model, idx_col, obj = L1_SVM_CG(X_train, 
                                                           y_train, 
                                                           idx_col, 
                                                           lam, 
                                                           tol, 
                                                           time_limit, 
                                                           _model, 
                                                           [], 
                                                           f, 
                                                           is_sparse=False, 
                                                           dict_nnz={})
    ed = time.time()
    _time = ed - st
    
    beta = np.zeros(P)
    beta[support] = _beta
    
    return obj, _time, beta, beta0
    
    
    
    
    
    
def use_correlation_CG(X_train, y_train, lam, n_features, tol=1e-2):
    
    
    f = open('trash.txt', 'w')
    idx_col, time_correl   = init_correlation(X_train, y_train, n_features, f)
    
    N, P = np.shape(X_train)
    st       = time.time()
    idx_col = random.sample(range(P), n_features)
    _model = 0
    time_limit = 10000
    
    [_beta,beta0], support, _time, _model, idx_col, obj = L1_SVM_CG(X_train, 
                                                           y_train, 
                                                           idx_col, 
                                                           lam, 
                                                           tol, 
                                                           time_limit, 
                                                           _model, 
                                                           [], 
                                                           f, 
                                                           is_sparse=False, 
                                                           dict_nnz={})
    ed = time.time()
    _time = ed - st
    
    beta = np.zeros(P)
    beta[support] = _beta
    
    return obj, _time, beta, beta0


    
