import numpy as np
from gurobipy import *
from L0_Slope_CG_model import *

from scipy.stats.stats import pearsonr 

import time
# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


def init_correlation(X_train, y_train, n_features, f):

#Run a two stpe RFE as suggested

#OUTPUT
#index_CG the
    
    start = time.time()
    N,P = X_train.shape

#---First RFE by removing half of the features at every iteration
    if(n_features<=P):
        correlations    = np.dot(X_train.T, y_train)       #class are balanced so we always can compute this way
        argsort_columns = np.argsort(np.abs(correlations))
        index_CG        = argsort_columns[::-1][:n_features]

    time_correl = time.time()-start
    write_and_print('Time correlation for column subset selection: '+str(time_correl), f)
    return index_CG.tolist(), time_correl









def L0_Slope_CG(X_train, y_train, w_star_FOM_Slope, index_CG, alpha, K_Slope, lambda_arr, use_lagrangian, delta, epsilon_RC, time_limit, model, warm_start, f, duality_gap=0):

 
    start = time.time()
    
    N,P      = X_train.shape
    aux_loop = 0   #count he number of rounds 
    aux_cuts = 1


#---Build the model
    model     = L0_Slope_CG_model(X_train, y_train, w_star_FOM_Slope, index_CG, alpha, K_Slope, use_lagrangian, delta, time_limit, model, warm_start, f) 

    columns_to_check = list(set(range(P))-set(index_CG))
    ones_P = np.ones(P)
    

#---Infinite loop until all the variables have non reduced cost
    continue_loop = True
    while continue_loop:
        continue_loop = False

        aux_loop += 1
        write_and_print('Round '+str(aux_loop), f)
        
    #---Solve the problem to get the dual solution
        model.optimize()
    
    #---Model
        dual_slacks       = [model.getConstrByName('slack_'+str(idx)) for idx in range(N)]
        dual_slack_values = [dual_slack.Pi for dual_slack in dual_slacks]

        betas_plus        = [model.getVarByName('beta_+_'+str(idx)) for idx in index_CG]
        betas_minus       = [model.getVarByName('beta_-_'+str(idx)) for idx in index_CG]
        beta              = np.array([beta_plus.X  for beta_plus  in betas_plus]) - np.array([beta_minus.X for beta_minus in betas_minus])
        
        b0                = model.getVarByName('b0')
        b0_value          = b0.X

        M                 = model.getVarByName('M')



#-------ADD CONSTRAINTS 
    #---Check if Slope condition
        idx_sort     = np.argsort(np.abs(beta))[::-1]#[-K_Slope:]
        w_star_Slope = np.zeros(len(index_CG))   
        for j in range(len(index_CG)): w_star_Slope[idx_sort[j]] = lambda_arr[j]
        
        RHS = delta if not use_lagrangian else M.X

        if np.sum([ abs(beta[idx_sort[j]]) * lambda_arr[j] for j in range(len(idx_sort)) ]) > (1+epsilon_RC)*RHS:
            continue_loop = True
            print(np.sum([ abs(beta[idx_sort[j]]) * lambda_arr[j] for j in range(len(idx_sort)) ]), RHS )

            RHS = delta if not use_lagrangian else M 
            
            model     = add_constraints_L0_Slope(X_train, y_train, model, w_star_Slope, RHS, betas_plus, betas_minus, b0, index_CG, aux_cuts)
            aux_cuts += 1





#-------ADD COLUMNS    
        all_w_star_cstrts = [model.getConstrByName('w_star_'+str(aux_cut)) for aux_cut in range(aux_cuts)]
        
        #support = np.where(beta!=0)[0]
        #print beta[support]
        #if aux_loop>1:
        #    donde = [idx in violated_columns for idx in index_CG]
        #    print beta[donde]


    #---Look for columns with negative reduced costs
        if len(columns_to_check) > 0:

            RC_aux             = np.array([y_train[i]*dual_slack_values[i] for i in range(N)])
            RC_array           = np.abs( np.dot(X_train.T, RC_aux) ) 
            RC_argsort         = np.argsort(RC_array)[::-1]
            all_col_ranks      = np.argsort(RC_argsort) #ranking of the features

            RC_array           = np.array([alpha*lambda_arr[len(index_CG)] - RC_array[j] for j in range(P)])[columns_to_check]
            RC_argsort         = np.argsort(RC_array)

            #RC_array           = np.array([alpha*np.sum(lambda_arr[: all_col_ranks[j]]) - cumsum_RC_array[all_col_ranks[j]] for j in range(P)])[columns_to_check]
            #RC_argsort         = np.argsort(RC_array)

            violated_columns = []
            violated_lambdas = []
            idx_add_col      = 0  


            while RC_array[RC_argsort[idx_add_col]] < - epsilon_RC   and   idx_add_col < min(10, len(index_CG), len(columns_to_check)) :
                violated_columns.append(np.array(columns_to_check)[RC_argsort[idx_add_col]])
                violated_lambdas.append(lambda_arr[len(index_CG) + idx_add_col] )
                idx_add_col += 1



        #---Add the columns with negative reduced costs
            n_columns_to_add = np.array(violated_columns).shape[0]

            if n_columns_to_add>0:
                continue_loop = True
                write_and_print('Number of columns added: '+str(n_columns_to_add), f)

            #---Take into accounts the previous Slope constraints -> otherwise coeffs to infinity
                model, new_betas_plus, new_betas_minus = add_columns_L0_Slope(X_train, y_train, model, violated_columns, violated_lambdas, range(N), dual_slacks, all_w_star_cstrts) 

                for i in range(n_columns_to_add):
                    column_to_add = violated_columns[i]

                    index_CG.append(column_to_add)
                    columns_to_check.remove(column_to_add)



        #---Add additional constraints for new coeffs
            #for j in range(len(idx_sort)): w_star_Slope[idx_sort[j]] = lambda_arr[len(violated_columns) + j]
            #print w_star_Slope

            #betas_plus  += new_betas_plus
            #betas_minus += new_betas_minus
            #model        = add_constraints_L0_Slope(X_train, y_train, model, w_star_Slope, RHS, betas_plus, betas_minus, b0, index_CG, aux_cuts)
            #aux_cuts    += 1




#---Solution
    beta_plus   = np.array([model.getVarByName('beta_+_'+str(idx)).X  for idx in index_CG])
    beta_minus  = np.array([model.getVarByName('beta_-_'+str(idx)).X  for idx in index_CG])
    beta    = np.array(beta_plus) - np.array(beta_minus)

    obj_val = model.ObjVal


#---TIME STOPS HERE
    time_CG = time.time()-start 
    write_and_print('\nTIME CG = '+str(time_CG), f)   
    


#---support and Objective value
    support = np.where(beta!=0)[0]
    beta    = beta[support]

    support = np.array(index_CG)[support]
    write_and_print('\nObj value   = '+str(obj_val), f)
    write_and_print('Len support = '+str(len(support)), f)

    b0   = model.getVarByName('b0').X 
    write_and_print('b0   = '+str(b0), f)


#---Violated constraints and dual support
    constraints = np.ones(N) - y_train*( X_train[:, support].dot(beta) + b0*np.ones(N))
    violated_constraints = np.arange(N)[constraints >= 0]
    write_and_print('\nNumber violated constraints =  '+str(violated_constraints.shape[0]), f)


    solution_dual = np.array([model.getConstrByName('slack_'+str(index)).Pi for index in range(N)])
    support_dual  = np.where(solution_dual!=0)[0]
    write_and_print('Len support dual = '+str(len(support_dual)), f)


    return [beta, b0], support, time_CG, model, index_CG, obj_val








#RC_aux             = np.array([y_train[i]*dual_slack_values[i] for i in range(N)])
#RC_array           = np.abs( np.dot(X_train.T, RC_aux) ) 
#RC_argsort         = np.argsort(RC_array)[::-1]

#all_col_ranks      = np.argsort(RC_argsort) #ranking of the features
#RC_array           = np.array([alpha*lambda_arr[all_col_ranks[j]] - RC_array[j] for j in range(P)])[columns_to_check]
#RC_argsort         = np.argsort(RC_array)





#cumsum_RC_array   = abs(RC_array[RC_argsort[0]])*np.ones(P)
#for j in range(1, P):
#    cumsum_RC_array[j] = cumsum_RC_array[j-1] + abs(RC_array[RC_argsort[j]])


#RC_array           = np.array([alpha*np.sum(lambda_arr[: all_col_ranks[j]]) - cumsum_RC_array[all_col_ranks[j]] for j in range(P)])[columns_to_check]
#RC_argsort         = np.argsort(RC_array)

