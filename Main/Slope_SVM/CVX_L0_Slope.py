# from cvxpy import *
#from cvxopt import *
import cvxpy as cp

import time
import numpy as np


# def CVX_L0_Slope(X_train, y_train, alpha, K_Slope, f, model):
# 	start_time = time.time()
# 	N, P = X_train.shape

# 	beta  = cp.Variable(P)
# 	beta0 = cp.Variable()
# 	loss  = cp.sum_entries(cp.pos(1 - cp.mul_elemwise(y_train, X_train*beta + beta0)))
# 	reg1  = cp.norm(beta, 1)

# 	#for j in range(min(P-1, K_max)): reg2  += (lambda_arr[j]-lambda_arr[j+1]) * sum_largest(abs(beta), j+1)  
# 	reg2 = cp.sum_largest(abs(beta), K_Slope) 

# 	prob  = cp.Problem(cp.Minimize(loss + alpha*reg1 + alpha*reg2))

# 	if model=='gurobi':
# 		prob.solve(solver=GUROBI)
# 	elif model=='ecos':
# 		prob.solve(solver=ECOS)
# 	elif model=='cvxopt':
# 		prob.solve(solver=CVXOPT)
	
# 	beta_cvx = np.array([ float(beta.value[i]) for i in range(P) ])
# 	end_time = time.time()-start_time
# 	write_and_print('\nTIME CVX = '+str(end_time), f)   
# 	support = np.where(np.round(beta_cvx,5) !=0)[0]
# 	write_and_print('Obj value   = '+str(prob.value), f)
# 	write_and_print('Len support = '+str(len(support)), f)

 
# 	constraints = np.ones(N) - y_train*( np.dot(X_train, beta_cvx) + beta0.value*np.ones(N))
# 	argsort     = np.argsort(np.abs(beta_cvx))[::-1]
	
# 	obj_val     = np.sum([max(constraints[i], 0) for i in range(N)]) + alpha*np.sum(np.abs(beta_cvx)) + alpha*np.sum(np.abs(beta_cvx[  argsort[:K_Slope] ])) 
# 	write_and_print('Obj value BIS  = '+str(obj_val), f)

# # 	return beta.value, prob.value, end_time
# 	return prob.value, end_time, beta.value, beta0.value




def CVX_L0_Slope(X_train, y_train, alpha, K_Slope, f, model):
    start_time = time.time()
    N, P = X_train.shape

    beta  = cp.Variable(P)
    beta0 = cp.Variable()
    loss  = cp.sum(cp.pos(1 - cp.multiply(y_train, X_train*beta + beta0)))
    reg1  = cp.norm(beta, 1)

    #for j in range(min(P-1, K_max)): reg2  += (lambda_arr[j]-lambda_arr[j+1]) * sum_largest(abs(beta), j+1)  
    reg2 = cp.sum_largest(cp.abs(beta), K_Slope) 

    prob  = cp.Problem(cp.Minimize(loss + alpha*reg1 + alpha*reg2))

    if model=='gurobi':
        prob.solve(solver='GUROBI', verbose=True)
    elif model=='ecos':
        prob.solve(solver='ECOS', verbose=True)
    elif model=='cvxopt':
        prob.solve(solver='CVXOPT', verbose=True)
    elif model=='scs':
        prob.solve(solver='SCS', verbose=True)

    beta_cvx = np.array([ float(beta.value[i]) for i in range(P) ])
    end_time = time.time()-start_time
    write_and_print('\nTIME CVX = '+str(end_time), f)   
    support = np.where(np.round(beta_cvx,5) !=0)[0]
    write_and_print('Obj value   = '+str(prob.value), f)
    write_and_print('Len support = '+str(len(support)), f)


    constraints = np.ones(N) - y_train*( np.dot(X_train, beta_cvx) + beta0.value*np.ones(N))
    argsort     = np.argsort(np.abs(beta_cvx))[::-1]

    obj_val     = np.sum([max(constraints[i], 0) for i in range(N)]) + alpha*np.sum(np.abs(beta_cvx)) + alpha*np.sum(np.abs(beta_cvx[  argsort[:K_Slope] ])) 
    write_and_print('Obj value BIS  = '+str(obj_val), f)

    # 	return beta.value, prob.value, end_time
    return prob.value, end_time, beta.value, beta0.value









def write_and_print(text,f):
    print(text)
    f.write('\n'+text)
