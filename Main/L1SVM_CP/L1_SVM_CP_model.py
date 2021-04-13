import numpy as np
from gurobipy import *

# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


#-----------------------------------------------BUILD THE MODEL--------------------------------------------------

def L1_SVM_CP_model(X, y, idx_CP, alpha, time_limit, model, warm_start, f):

#idx_CP    = index of the constraints used to generate 
#MODEL     = previous model to speed up
    

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P  = X.shape
    N_CP = len(idx_CP)

    write_and_print('Size of model: '+str([N_CP, P]), f)

    
    if(model == 0):
    
    #---VARIABLES
        L1_SVM_CP=Model("L1_SVM_CP")
        L1_SVM_CP.setParam('OutputFlag', False )
        L1_SVM_CP.setParam('TimeLimit', time_limit)
        
        
        #Hinge loss
        xi = np.array([L1_SVM_CP.addVar(lb=0, name="loss_"+str(idx_CP[i])) for i in range(N_CP)])

        #Beta -> name correspond to real index
        beta_plus  = np.array([L1_SVM_CP.addVar(lb=0, name="beta_+_"+str(i)) for i in range(P)])
        beta_minus = np.array([L1_SVM_CP.addVar(lb=0, name="beta_-_"+str(i)) for i in range(P)])
        b0 = L1_SVM_CP.addVar(lb=-GRB.INFINITY, name="b0")
        
        L1_SVM_CP.update()


    #---OBJECTIVE VALUE 
        L1_SVM_CP.setObjective(quicksum(xi) + alpha*quicksum(beta_plus[i]+beta_minus[i] for i in range(P)), GRB.MINIMIZE)


    #---HIGE CONSTRAINTS ONLY FOR SUBSET
        for i in range(N_CP):
            L1_SVM_CP.addConstr(xi[i] + y[idx_CP[i]]*(b0 + quicksum([ X[idx_CP[i]][k]*(beta_plus[k] - beta_minus[k]) for k in range(P)]))>= 1, 
                                 name="slack_"+str(idx_CP[i]))
      
        L1_SVM_CP.update()



    #---POSSIBLE WARM START (only for Gurobi on full model)
        if(len(warm_start) > 0):

            for i in range(P):
                beta_plus[i].start  = max( warm_start[i], 0)
                beta_minus[i].start = max(-warm_start[i], 0)
        
    
    
        
#---IF PREVIOUS MODEL JUST UPDATE THE PENALIZATION
    else:
        L1_SVM_CP = model.copy()
        
        xi          = [L1_SVM_CP.getVarByName('loss_'+str(idx_CP[i])) for i in range(N_CP)]
        beta_plus   = [L1_SVM_CP.getVarByName('beta_+_'+str(i)) for i in range(P)]
        beta_minus  = [L1_SVM_CP.getVarByName('beta_-_'+str(i)) for i in range(P)]

        L1_SVM_CP.setObjective(quicksum(xi) + alpha*quicksum(beta_plus[i]+beta_minus[i] for i in range(P)), GRB.MINIMIZE)
      

    L1_SVM_CP.update()
    
    return L1_SVM_CP







#-----------------------------------------------ADD VARIABLES--------------------------------------------------

def add_constraints_L1_SVM(X, y, L1_SVM_CP, violated_constraints, beta_plus, beta_minus, b0, idx_columns, is_sparse=False, dict_nnz={}):
# Add a column in a model with respective constraints

    for violated_constraint in violated_constraints:

        xi_violated = L1_SVM_CP.addVar(lb=0, obj = 1, column=Column(),  name="loss_"+str(violated_constraint) )
        L1_SVM_CP.update()

        if not is_sparse:
            L1_SVM_CP.addConstr(  xi_violated + y[violated_constraint]*(b0 + quicksum([ X[violated_constraint][idx_columns[k]]*(beta_plus[k] - beta_minus[k]) for k in range(len(idx_columns))]))>= 1, name="slack_"+str(violated_constraint))
        else:
            inter_index_columns_nnz_i = list( set(idx_columns) & set(dict_nnz[ violated_constraint ]) )
            indexes_coeffs            = [idx_columns.index(inter_index) for inter_index in inter_index_columns_nnz_i]
            ### CAREFUll X[i,j]
            L1_SVM_CP.addConstr( xi_violated + y[violated_constraint]*(b0 + quicksum([ X[violated_constraint, inter_index_columns_nnz_i[k]]*(beta_plus[indexes_coeffs[k]] - beta_minus[indexes_coeffs[k]]) for k in range(len(inter_index_columns_nnz_i)) ]))>= 1, name="slack_"+str(violated_constraint))

    L1_SVM_CP.update()    

    return L1_SVM_CP





