
import cvxpy as cp
from sklearn.linear_model import SGDClassifier
from smoothing_hinge_loss import *
import time

import os
# os.environ["GUROBI_HOME"]="/home/software/gurobi/gurobi811/linux64/"
# os.environ["GRB_LICENSE_FILE"]="/home/software/gurobi/gurobi.lic"

from L1_SVM_CG_model import *
from L1_SVM_both_CG_CP import *
from L1_SVM_add_columns_delete_samples import *
from init_L1_SVM_both_CG_CP import *

from smoothing_hinge_loss import *

# sys.path.append('../real_data')
# from process_real_datasets import *
# from read_real_datasets    import *




def use_FOM_CGCP(X_train, y_train, lam, relative_lam, tol=1e-2, is_sparse = False, dict_nnz={}):
    f = open('trash.txt', 'w')
    rho = relative_lam
    st = time.time()
    N,P = np.shape(X_train)
    time_limit = 3600
    idx_samples, idx_cols, time_FOM = init_both_CG_CP_sampling_smoothing(X_train, y_train, 
                                                                  lam, rho, f, 
                                                                  is_sparse=is_sparse)	
    
    ed1 = time.time()
    model_SVM_CCG = 0
    beta_, beta0, support, time_, model_, idx_samples, idx_cols, obj_ = L1_SVM_both_CG_CP(X_train, y_train, 
                                                                                    idx_samples, idx_cols, 
                                                                                    lam, tol, 
                                                                                    time_limit, model_SVM_CCG, 
                                                                                    [], f, 
                                                                                    is_sparse=is_sparse, dict_nnz=dict_nnz)
    beta = np.zeros(P)
    beta[idx_cols] = beta_
    ed2 = time.time()
    
    time_FOM = ed1 - st
    time_CGCP = ed2 - ed1
    time_total = ed2 - st
    
    
    constraints = np.ones(N) - y_train*( X_train[:, idx_cols].dot(beta_) + beta0*np.ones(N))
#     constraints = np.ones(N) - y_train * (np.dot(X_train, beta) + beta0) 
    obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_))


    
    return obj, time_total, time_CGCP, beta, beta0




# def use_CGCP(X_train, y_train, lam, dict_nnz):
#     f = open('trash.txt', 'w')
#     st = time.time()
#     time_limit = 3600

#     idx_samples = [0]
#     idx_cols = [0]
#     model_SVM_CCG = 0
#     beta, beta0, support, time_, model_, idx_samples, idx_cols, obj = L1_SVM_both_CG_CP(X_train, y_train, 
#                                                                                     idx_samples, idx_cols, 
#                                                                                     lam, 1e-2, 
#                                                                                     time_limit, model_SVM_CCG, 
#                                                                                     [], f, 
#                                                                                     is_sparse=True, dict_nnz=dict_nnz)
    
#     ed = time.time()
    
#     return obj, time_, beta, beta0

