import numpy as np
from gurobipy import *

# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


#-----------------------------------------------BUILD THE MODEL--------------------------------------------------

def L1_SVM_both_CG_CP_model(X, y, index_samples, index_columns, alpha, time_limit, L1_SVM_CG_CP, warm_start, f, is_sparse=False, dict_nnz={}):

#idx_CG    = index of the features used to generate 
#MODEL     = previous model to speed up
    

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P  = X.shape
    N_CP = len(index_samples)
    P_CG = len(index_columns)

    write_and_print('Size of model: '+str([N_CP, P_CG]), f)


    
    if(L1_SVM_CG_CP == 0):
    
    #---VARIABLES
        L1_SVM_CG_CP=Model("L1_SVM_CG_CP")
        L1_SVM_CG_CP.setParam('OutputFlag', True )
        L1_SVM_CG_CP.setParam('TimeLimit', time_limit)
        
        
        #Hinge loss
        xi = np.array([L1_SVM_CG_CP.addVar(lb=0, name="loss_"+str(index_samples[i])) for i in range(N_CP)])

        #Beta -> name correspond to real index
        beta_plus  = np.array([L1_SVM_CG_CP.addVar(lb=0, name="beta_+_"+str(index_columns[i])) for i in range(P_CG)])
        beta_minus = np.array([L1_SVM_CG_CP.addVar(lb=0, name="beta_-_"+str(index_columns[i])) for i in range(P_CG)])
        b0 = L1_SVM_CG_CP.addVar(lb=-GRB.INFINITY, name="b0")
        
        L1_SVM_CG_CP.update()


    #---OBJECTIVE VALUE 
        L1_SVM_CG_CP.setObjective(quicksum(xi) + alpha*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)


    #---HIGE CONSTRAINTS
        for i in range(N_CP):
            if not is_sparse:
                L1_SVM_CG_CP.addConstr(xi[i] + y[index_samples[i]]*(b0 + quicksum([ X[index_samples[i]][index_columns[k]]*(beta_plus[k] - beta_minus[k]) for k in range(P_CG)]))>= 1, name="slack_"+str(index_samples[i]))
            else:
                inter_index_columns_nnz_i = list( set(index_columns) & set(dict_nnz[ index_samples[i] ]) )
                indexes_coeffs            = [index_columns.index(inter_index) for inter_index in inter_index_columns_nnz_i]
                ### CAREFUll X[i,j]
                L1_SVM_CG_CP.addConstr(xi[i] + y[index_samples[i]]*(b0 + quicksum([ X[index_samples[i], inter_index_columns_nnz_i[k]]*(beta_plus[indexes_coeffs[k]] - beta_minus[indexes_coeffs[k]]) for k in range(len(inter_index_columns_nnz_i)) ]))>= 1, name="slack_"+str(index_samples[i]))
            
    #---RELAX
        L1_SVM_CG_CP.update()


    #---POSSIBLE WARM START (only for Gurobi on full model)
        if(len(warm_start) > 0):
            print("WARM START")
            
            for i in range(P_CG):
                beta_plus[i].start  = max( warm_start[index_columns[i]], 0)
                beta_minus[i].start = max(-warm_start[index_columns[i]], 0)
        
    
    
        
#---IF PREVIOUS MODEL JUST UPDATE THE PENALIZATION
    else:
        xi          = [L1_SVM_CG_CP.getVarByName('loss_'+str(index_samples[i]))   for i in range(N_CP)]
        beta_plus   = [L1_SVM_CG_CP.getVarByName('beta_+_'+str(index_columns[i])) for i in range(P_CG)]
        beta_minus  = [L1_SVM_CG_CP.getVarByName('beta_-_'+str(index_columns[i])) for i in range(P_CG)]

        L1_SVM_CG_CP.setObjective(quicksum(xi) + alpha*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)
   

    L1_SVM_CG_CP.update()
    
    return L1_SVM_CG_CP








