import numpy as np
from gurobipy import *

import time
# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *

sys.path.append('../L1SVM_CG')
from L1_SVM_CG_model import *

sys.path.append('../L1SVM_CP')
from L1_SVM_CP_model import *






def L1_SVM_add_columns_delete_samples(X_train, y_train, index_samples, index_columns, alpha, epsilon_RC, time_limit, model, delete_samples, f):


#INPUT
#index_samples : index of samples currently in model
#index_columns : index of columns currently in model


## USED TO COMPUTE THE PATH
# At every iteration we train the restricted model, check if



    N,P = X_train.shape
    aux = 0   #count he number of rounds 


#---Build the model
    start = time.time()
    model = L1_SVM_CG_model(X_train, y_train, index_columns, alpha, time_limit, model, f) #model=0 -> start with all samples but small subset of collumns 
                                                                         #else update the objective function
    is_L1_SVM = (len(index_columns) == N) and (len(index_columns) == P)
    


#---Infinite loop until all the variables have non reduced cost and all 
    continue_loop = True
    

    while continue_loop:
        continue_loop = False
        aux += 1
        write_and_print('Round '+str(aux), f)

        model.optimize()
        write_and_print('Time optimizing = '+str(time.time()-start), f)
        


    #---Usefull model chracteristics
        dual_slack = [model.getConstrByName('slack_'+str(idx)).Pi for idx in index_samples]


    #---Usefull model chracteristics
        beta_plus  = np.zeros(P) 
        beta_minus = np.zeros(P) 

        for idx in index_columns:
            beta_plus[idx]  = model.getVarByName('beta_+_'+str(idx)).X 
            beta_minus[idx] = model.getVarByName('beta_-_'+str(idx)).X 
        
        beta = beta_plus - beta_minus
        b0   = model.getVarByName('b0').X




#-------REDUCED COSTS

    #---Look for variable with negative reduced costs
        violated_columns = []
        
        for column in set(range(P))-set(index_columns): 
        #---DOESNT HAPPEN FOR L1 SVM
            reduced_cost = np.sum([y_train[i]*X_train[i,column]*dual_slack[i] for i in range(len(index_samples))])
            reduced_cost = alpha  + min(reduced_cost, -reduced_cost)

            if reduced_cost < -epsilon_RC:
                violated_columns.append(column)
                 
    #---Add the column with negative reduced costs
        n_columns_to_add = len(violated_columns)

        if n_columns_to_add>0:
            continue_loop = True
            
            write_and_print('Number of columns added: '+str(n_columns_to_add), f)

            for i in range(n_columns_to_add):
                column_to_add = violated_columns[i]
                model = add_column_L1_SVM(X_train, y_train, model, column_to_add, index_samples, alpha) 
                model.update()

                index_columns.append(column_to_add)

            #model.optimize()





#-------VIOLATED CONSTRAINTS COSTS

    #---Compute all reduce cost and look for variable with negative reduced costs
        violated_constraints     = []
        most_violated_constraint = -1
        most_violated_cost       = 0
        
        
        for constraint in set(range(N))-set(index_samples):
        #---DOESNT HAPPEN FOR L1 SVM
            constraint_value = 1 - y_train[constraint]*(np.dot(X_train[constraint,:], beta) + b0)
            if constraint_value > epsilon_RC:
                violated_constraints.append(constraint)
                
    #---Add the column with the most most violated constraint to the original model (not the relax !!)
        n_constraints_to_add = len(violated_constraints)

        if n_constraints_to_add>0:
            continue_loop = True
            
            write_and_print('Number of constraints added: '+str(n_constraints_to_add), f)
            model = add_constraints_L1_SVM(X_train, y_train, model, violated_constraints, index_columns) 
            model.update()

            for violated_constraint in violated_constraints:
                index_samples.append(violated_constraint)




   



#---Solution
    beta_plus   = np.array([model.getVarByName('beta_+_'+str(idx)).X  for idx in index_columns])
    beta_minus  = np.array([model.getVarByName('beta_-_'+str(idx)).X  for idx in index_columns])

    beta    = np.round(beta_plus - beta_minus,6)
    support = np.where(beta!=0)[0]

    objval = model.ObjVal
    write_and_print('\nObj value   = '+str(objval), f)
    write_and_print('\nLen support = '+str(len(support)), f)


    solution_dual = np.array([model.getConstrByName('slack_'+str(idx)).Pi for idx in index_samples])
    support_dual  = np.where(solution_dual!=0)[0]
    write_and_print('Len support dual = '+str(len(support_dual)), f)


#---Stop deleting features?
    delete_samples = (len(support_dual) > 1.5*len(support))




#---If lambda decrease, delete samples not in support of dual
    if delete_samples and not is_L1_SVM: #CANNOT HAPPEN FOR L1 SVM

        idx_to_removes   = np.array(index_samples)[solution_dual==0] #non in dual solution
        slacks_to_remove = np.array([model.getConstrByName('slack_'+str(idx)) for idx in idx_to_removes ]) 

        for idx_to_remove in idx_to_removes:
            index_samples.remove(idx_to_remove)

        for slack_to_remove in slacks_to_remove:
            model.remove(slack_to_remove)
        model.update()
        
    
    time_add_columns_delete_samples = time.time()-start 


#---End
    write_and_print('Time = '+str(time_add_columns_delete_samples), f)
    return beta, support, time_add_columns_delete_samples, model, index_samples, index_columns, delete_samples, objval









