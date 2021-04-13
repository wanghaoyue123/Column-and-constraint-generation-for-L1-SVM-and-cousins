import numpy as np
from gurobipy import *
from group_Linf_SVM_CG_model import *

from scipy.stats.stats import pearsonr 

import time
# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


#-----------------------------------------------INITIALIZE WITH RFE--------------------------------------------------





def init_group_Linf(X_train, y_train, group_to_feat, n_groups, f):
	
	start = time.time()
	N,P = X_train.shape

	if n_groups <= group_to_feat.shape[0]:
		abs_correlations      = np.abs(np.dot(X_train.T, y_train))       #class are balanced so we always can compute this way
		sum_abs_correl_groups = [np.sum(abs_correlations[idx]) for idx in group_to_feat]

		argsort_columns = np.argsort(sum_abs_correl_groups)
		index_CG        = argsort_columns[::-1][:n_groups]

	time_correl = time.time()-start
	write_and_print('Time correlation for group selection: '+str(time_correl), f)
	return index_CG.tolist(), time_correl






def group_Linf_SVM_CG(X_train, y_train, group_to_feat, idx_groups_CG, alpha, epsilon_RC, time_limit, model, warm_start, f, duality_gap=0):


#INPUT
#n_features_RFE : number of features to give to RFE to intialize
#epsilon_RC     : maximum non negatuve reduced cost
	
	start = time.time()
	
	N,P = X_train.shape
	aux = 0   #count he number of rounds 


#---Build the model
	model = group_Linf_SVM_CG_model(X_train, y_train, group_to_feat, idx_groups_CG, alpha, time_limit, model, warm_start, f, duality_gap=duality_gap) #model=0 -> no warm start else update the objective function
	
	number_groups     = group_to_feat.shape[0]
	is_group_Linf_SVM = (len(idx_groups_CG) == number_groups)
	
	groups_to_check = list(set(range(number_groups))-set(idx_groups_CG))
	ones_NG = np.ones(number_groups)
	


#---Infinite loop until all the variables have non reduced cost
	while True:
		aux += 1
		write_and_print('Round '+str(aux), f)
		
	#---Solve the problem to get the dual solution
		model.optimize()
		write_and_print('Time optimizing = '+str(time.time()-start), f)

		if not is_group_Linf_SVM:

		#---Compute all reduce cost and look for variable with negative reduced costs
			dual_slacks      = [model.getConstrByName('slack_'+str(i)) for i in range(N)]
			dual_slack_values= [dual_slack.Pi for dual_slack in dual_slacks]
			
			RC_array = []
			RC_aux   = np.array([y_train[i]*dual_slack_values[i] for i in range(N)])

			for idx in groups_to_check:
				RC_array.append(alpha - np.sum( np.abs(np.dot(X_train[:, np.array(group_to_feat[idx])].T, RC_aux))) )

			idx_groups_to_add= np.array(groups_to_check)[np.array(RC_array) < -epsilon_RC]
					   

		#---Add the groups with negative reduced costs
			n_groups_to_add = idx_groups_to_add.shape[0]

			if n_groups_to_add>0:
				write_and_print('Number of groups added: '+str(n_groups_to_add), f)

				model = add_groups_Linf_SVM(X_train, y_train, group_to_feat, model, idx_groups_to_add, range(N), dual_slacks, alpha) 

				for i in range(n_groups_to_add):
					group_to_add = idx_groups_to_add[i]
					idx_groups_CG.append(group_to_add)
					groups_to_check.remove(group_to_add)
			else:
				break

			
	#---WE ALWAYS BREAK FOR group_Linf SVM   
		else:
			break 



#---Solution
	beta_plus  = []
	beta_minus = []

	for idx in idx_groups_CG:
		try:
			beta_plus   += [model.getVarByName('beta_+_'+str(j)).X  for j in group_to_feat[idx]]
			beta_minus  += [model.getVarByName('beta_-_'+str(j)).X  for j in group_to_feat[idx]]
		except:
			None
	beta = np.array(beta_plus) - np.array(beta_minus)

	obj_val = model.ObjVal


#---TIME STOPS HERE
	time_CG = time.time()-start 
	write_and_print('\nTIME CG = '+str(time_CG), f)   
	

#---Support and Objective value
	support = np.where(beta!=0)[0]
	beta    = beta[support]

	#print np.array([group_to_feat[idx] for idx in idx_groups_CG])




########### SOUCI ###########


	#support = np.array([group_to_feat[idx] for idx in idx_groups_CG])[support]
	write_and_print('\nObj value   = '+str(obj_val), f)
	write_and_print('Len support = '+str(len(support)), f)

	try:
		b0   = model.getVarByName('b0').X 
		write_and_print('b0   = '+str(b0), f)


	#---Violated constraints and dual support
		constraints = np.ones(N) - y_train*( np.dot(X_train[:, support], beta) + b0*np.ones(N))
		violated_constraints = np.arange(N)[constraints >= 0]
		write_and_print('\nNumber violated constraints =  '+str(violated_constraints.shape[0]), f)


		solution_dual = np.array([model.getConstrByName('slack_'+str(index)).Pi for index in range(N)])
		support_dual  = np.where(solution_dual!=0)[0]
		write_and_print('Len support dual = '+str(len(support_dual)), f)
	except:
		b0 = 0

	return [beta, b0], support, time_CG, model, idx_groups_CG, obj_val









