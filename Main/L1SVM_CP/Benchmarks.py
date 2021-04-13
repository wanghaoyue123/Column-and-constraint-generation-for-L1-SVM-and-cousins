
import cvxpy as cp
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
ro.r('''library(fastclime)''')

from sklearn.linear_model import SGDClassifier
from smoothing_hinge_loss import *
import time

import os
# os.environ["GUROBI_HOME"]="/home/software/gurobi/gurobi811/linux64/"
# os.environ["GRB_LICENSE_FILE"]="/home/software/gurobi/gurobi.lic"




def use_SCS(X_train, y_train, lam):
    N,P = np.shape(X_train)
    beta = cp.Variable((P,1))
    beta0 = cp.Variable()
    y_train_ = np.reshape(y_train,(N,1))
    loss = cp.sum(cp.pos(1 - cp.multiply(y_train_, X_train*beta + beta0)))
    reg = cp.norm(beta, 1)
    prob = cp.Problem(cp.Minimize(loss + lam*reg))
    st = time.time()
    SCS_obj = prob.solve(solver = 'SCS', verbose=True)
    ed = time.time()
    SCS_time = ed - st
    
    beta_ = np.reshape(beta.value, (P,))
    beta0_ = beta0.value
    constraints = np.ones(N) - y_train * (np.dot(X_train, beta_) + beta0_) 
    SCS_obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_))
    
    
    return SCS_obj, SCS_time, beta_, beta0_





def use_PSM(X_train, y_train, lam):
    
    N,P = np.shape(X_train)
    
    mat = np.zeros((N, N+2*P+2))
    mat[:, 0:N] = -np.eye(N)
    mat[:, N:N+P] = - np.reshape(y_train, (N,1))*X_train
    mat[:, N+P:N+2*P] = np.reshape(y_train, (N,1))*X_train
    mat[:, N+2*P] = -y_train
    mat[:, N+2*P+1] = y_train
    rhs   = -np.ones(N)
    obj   = np.concatenate([-np.ones(N), -lam*np.ones(2*P), np.zeros(2)], axis=0)   
    
    ro.r.assign("mat_R", mat)
    ro.r.assign("rhs_R", rhs)
    ro.r.assign("obj_R", obj)
    ro.r.assign("lambda_R", lam)
    st = time.time()
    ro.r('''sol_R = fastlp(obj_R, mat_R, rhs_R)''')
    ed = time.time()
    PSM_time = ed - st
    
    
    sol = np.array(ro.r('''sol_R'''))
    xi, beta_plus, beta_minus, b0_plus, b0_minus = sol[:N], sol[N: N+P], sol[N+P: N+2*P], sol[-2], sol[-1]
    beta = beta_plus - beta_minus
    b0   = b0_plus   - b0_minus
    constraints = np.ones(N) - y_train*(np.dot(X_train, beta) + b0*np.ones(N)) 
    PSM_obj   = np.sum([max(constraints[i], 0) for i in range(N)]) + lam*np.sum(np.abs(beta))
    
    return PSM_obj, PSM_time, beta, b0







# def use_SGD(X_train, y_train, lam):
# 	N = X_train.shape[0]

# 	# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# 	clf = SGDClassifier(loss="hinge", 
# 						penalty="l1", 
# 						alpha=lam / float(N), 
# 						fit_intercept=True,
# 						max_iter=1e4,
# 						tol=None,
# 						learning_rate='optimal')

# 	start   = time.time()
# 	clf.fit(X_train, y_train)
# 	SGD_time = time.time() - start

# 	beta_SGD = clf.coef_[0]
# 	b0_SGD = clf.intercept_[0]
# 	constraints = np.ones(N) - y_train * (np.dot(X_train, beta_SGD) + b0_SGD) 
# 	SGD_obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_SGD))

# 	return SGD_obj, SGD_time, beta_SGD, b0_SGD





def use_SGD(X_train, y_train, lam, max_iter):
	N = X_train.shape[0]

	# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
	clf = SGDClassifier(loss="hinge", 
						penalty="l1", 
						alpha=lam / float(N), 
						fit_intercept=True,
						max_iter=max_iter,
						tol=None,
						learning_rate='optimal')

	start   = time.time()
	clf.fit(X_train, y_train)
	SGD_time = time.time() - start

	beta_SGD = clf.coef_[0]
	b0_SGD = clf.intercept_[0]
	constraints = np.ones(N) - y_train * (np.dot(X_train, beta_SGD) + b0_SGD) 
	SGD_obj = np.sum([max(constraints[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta_SGD))

	return SGD_obj, SGD_time, beta_SGD, b0_SGD






def use_FOM(X_train, y_train, lam, tau, max_iter):
    N,P = np.shape(X_train)
    f = open('trash.txt', 'w')
    n_iter  = 1e5
    n_loop  = 1
    _, _, FOM_time, beta, beta0 = loop_smoothing_hinge_loss('hinge', 
                                                      'l1', 
                                                      X_train, 
                                                      y_train, 
                                                      lam, 
                                                      tau, 
                                                      n_loop, 
                                                      n_iter, 
                                                      f, 
                                                      is_sparse=False)
    beta0 = beta0/np.sqrt(N)
    cons = np.ones(N) - y_train*( np.dot(X_train, beta) + beta0) 
    FOM_obj = np.sum([max(cons[i], 0) for i in range(N)]) + lam * np.sum(np.abs(beta))
    
    return FOM_obj, FOM_time, beta, beta0





def use_Gurobi(X_train, y_train, lam):
    N,P = np.shape(X_train)
    beta = cp.Variable((P,1))
    beta0 = cp.Variable()
    y_train_ = np.reshape(y_train,(N,1))
    loss = cp.sum(cp.pos(1 - cp.multiply(y_train_, X_train*beta + beta0)))
    reg = cp.norm(beta, 1)
    prob = cp.Problem(cp.Minimize(loss + lam*reg))
    st = time.time()
    gurobi_obj = prob.solve(solver = 'GUROBI', verbose=True)
    ed = time.time()
    gurobi_time = ed - st
    
    return gurobi_obj, gurobi_time, np.reshape(beta.value, (P,)), beta0.value


