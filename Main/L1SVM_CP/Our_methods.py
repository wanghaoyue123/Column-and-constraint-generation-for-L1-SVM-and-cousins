import os
# os.environ["GUROBI_HOME"]="/home/software/gurobi/gurobi811/linux64/"
# os.environ["GRB_LICENSE_FILE"]="/home/software/gurobi/gurobi.lic"

import numpy as np
import datetime
import time
import os
import sys
import math
import subprocess
import random
from gurobipy import *


from L1_SVM_CG import *
from L1_SVM_CP import *
from init_L1_SVM_CP import *
from smoothing_hinge_loss import *
from simulate_data_classification import *
from scipy.sparse import *
import random



def use_FOM_CP(X_train, y_train, lam, relative_lam, tol=1e-2):
    N,P = np.shape(X_train)
    f = open('trash.txt', 'w')
    rho = relative_lam
    _model = 0
    time_limit = 3600
    st = time.time()
    ## FOM
    index_samples_FOM, time_FOM = init_CP_sampling_smoothing(X_train, y_train, lam, rho, True, f)
    ed1 = time.time()
    ## CP
    beta, beta0, support, _time, _model, idx_cols, obj_ = L1_SVM_CP(X_train, y_train, 
                                                           index_samples_FOM, lam, 
                                                           tol, time_limit, 
                                                           _model, [], f)
    
    constraints = np.ones(N) - y_train * (np.dot(X_train, beta) + beta0) 
    obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta))
    
    
    ed2 = time.time()
    tm_CP = ed2 - ed1
    tm_total = ed2 - st

    return obj, tm_total, tm_CP, beta, beta0

