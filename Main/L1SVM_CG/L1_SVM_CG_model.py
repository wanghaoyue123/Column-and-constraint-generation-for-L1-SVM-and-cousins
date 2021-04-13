import numpy as np
from gurobipy import *

# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


#-----------------------------------------------BUILD THE MODEL--------------------------------------------------

def L1_SVM_CG_model(X, y, idx_CG, alpha, time_limit, L1_SVM_CG, warm_start, f, duality_gap=0, is_sparse=False, dict_nnz={}):

#idx_CG    = index of the features used to generate 
#L1_SVM_CG = previous model to speed up
#WARM START= speed up Gurobi on full model 
    

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P  = X.shape
    P_CG = len(idx_CG)

    write_and_print('Size of model: '+str([N, P_CG]), f)

    
    if(L1_SVM_CG == 0):
    
    #---VARIABLES
        L1_SVM_CG=Model("L1_SVM_CG")
        L1_SVM_CG.setParam('OutputFlag', False)
        #L1_SVM_CG.setParam('TimeLimit', time_limit)
        if duality_gap>0: L1_SVM_CG.Params.BarConvTol = duality_gap
        
        
        #Hinge loss
        xi = np.array([L1_SVM_CG.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])

        #Beta -> name correspond to real index
        beta_plus  = np.array([L1_SVM_CG.addVar(lb=0, name="beta_+_"+str(idx_CG[i])) for i in range(P_CG)])
        beta_minus = np.array([L1_SVM_CG.addVar(lb=0, name="beta_-_"+str(idx_CG[i])) for i in range(P_CG)])
        b0 = L1_SVM_CG.addVar(lb=-GRB.INFINITY, name="b0")

        L1_SVM_CG.update()


    #---OBJECTIVE VALUE 
        L1_SVM_CG.setObjective(quicksum(xi) + alpha*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)


    #---HIGE CONSTRAINTS
        for i in range(N):
            if not is_sparse:
                L1_SVM_CG.addConstr(xi[i] + y[i]*(b0 + quicksum([ X[i][idx_CG[k]]*(beta_plus[k] - beta_minus[k]) for k in range(P_CG)]))>= 1, name="slack_"+str(i))
            else:
                inter_index_columns_nnz_i = list( set(idx_CG) & set(dict_nnz[i]) )
                indexes_coeffs            = [idx_CG.index(inter_index) for inter_index in inter_index_columns_nnz_i]
                ### CAREFUll X[i,j]
                L1_SVM_CG.addConstr(xi[i] + y[i]*(b0 + quicksum([ X[i, inter_index_columns_nnz_i[k]]*(beta_plus[indexes_coeffs[k]] - beta_minus[indexes_coeffs[k]]) for k in range(len(inter_index_columns_nnz_i)) ]))>= 1, name="slack_"+str(i))


    #---POSSIBLE WARM START (only for Gurobi on full model)
        if(len(warm_start) > 0):

            for i in range(P_CG):
                beta_plus[i].start  = max( warm_start[idx_CG[i]], 0)
                beta_minus[i].start = max(-warm_start[idx_CG[i]], 0)


           
        
    
    
        
#---IF PREVIOUS MODEL JUST UPDATE THE PENALIZATION
    else:
        #L1_SVM_CG = model.copy()
        xi          = [L1_SVM_CG.getVarByName('loss_'+str(i)) for i in range(N)]
        beta_plus   = [L1_SVM_CG.getVarByName('beta_+_'+str(idx_CG[i])) for i in range(P_CG)]
        beta_minus  = [L1_SVM_CG.getVarByName('beta_-_'+str(idx_CG[i])) for i in range(P_CG)]

        L1_SVM_CG.setObjective(quicksum(xi) + alpha*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)

    #print L1_SVM_CG.Params
    return L1_SVM_CG







#-----------------------------------------------ADD VARIABLES--------------------------------------------------

def add_columns_L1_SVM(X, y, L1_SVM_CG, violated_columns, idx_samples, dual_slacks, alpha, is_sparse=False):

#idx_samples: samples in the model -> may be different from range(N) when deleting samples

    
    for violated_column in violated_columns:

        col_plus, col_minus = Column(), Column()

        for i in range(len(idx_samples)):
            X_row_col = X[idx_samples[i]][violated_column] if not is_sparse else X[idx_samples[i], violated_column] 
                
            if X_row_col != 0:
                col_plus.addTerms(  y[idx_samples[i]]*X_row_col, dual_slacks[i])
                col_minus.addTerms(-y[idx_samples[i]]*X_row_col, dual_slacks[i])
            
            
        beta_plus  = L1_SVM_CG.addVar(lb=0, obj = alpha, column=col_plus,  name="beta_+_"+str(violated_column) )
        beta_minus = L1_SVM_CG.addVar(lb=0, obj = alpha, column=col_minus, name="beta_-_"+str(violated_column) )

    L1_SVM_CG.update()

    return L1_SVM_CG





