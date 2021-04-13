import numpy as np
from gurobipy import *

# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *
import math

#-----------------------------------------------BUILD THE MODEL--------------------------------------------------

def L0_Slope_CG_model(X, y, w_star, idx_CG, alpha, K_Slope, use_lagrangian, delta, time_limit, L0_Slope_CG, warm_start, f, duality_gap=0):

#idx_CG    = index of the features used to generate 
#L0_Slope_CG = previous model to speed up
#WARM START= speed up Gurobi on full model 
    

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P  = X.shape
    P_CG = len(idx_CG)

    write_and_print('Size of model: '+str([N, P_CG]), f)

    
    if(L0_Slope_CG == 0):
    
    #---VARIABLES
        L0_Slope_CG=Model("L0_Slope_CG")
        L0_Slope_CG.setParam('OutputFlag', False)
        #L0_Slope_CG.setParam('TimeLimit', time_limit)
        if duality_gap>0: L0_Slope_CG.Params.BarConvTol = duality_gap
        
        
        #Hinge loss
        xi = np.array([L0_Slope_CG.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])

        #Beta 
        beta_plus  = np.array([L0_Slope_CG.addVar(lb=0, name="beta_+_"+str(idx_CG[i])) for i in range(P_CG)])
        beta_minus = np.array([L0_Slope_CG.addVar(lb=0, name="beta_-_"+str(idx_CG[i])) for i in range(P_CG)])
        b0 = L0_Slope_CG.addVar(lb=-GRB.INFINITY, name="b0")


        if use_lagrangian: M = L0_Slope_CG.addVar(lb=-GRB.INFINITY, name="M")
        L0_Slope_CG.update()


    #---OBJECTIVE VALUE 
        if not use_lagrangian:
            L0_Slope_CG.setObjective(quicksum(xi) + alpha*quicksum([ beta_plus[k] + beta_minus[k] for k in range(P_CG)]), GRB.MINIMIZE)
        else:
            L0_Slope_CG.setObjective(quicksum(xi) + alpha*M, GRB.MINIMIZE)


    #---HIGE CONSTRAINTS
        for i in range(N): L0_Slope_CG.addConstr(xi[i] + y[i]*(b0 + quicksum([ X[i][idx_CG[k]]*(beta_plus[k] - beta_minus[k]) for k in range(P_CG)]))>= 1, name="slack_"+str(i))


    #---SlOPE CONSTRAINTS
        if not use_lagrangian:
            L0_Slope_CG.addConstr( quicksum([ w_star[k]*(beta_plus[k] + beta_minus[k]) for k in range(P_CG)]) <= delta, name="w_star_0")
        else:
            L0_Slope_CG.addConstr( quicksum([ w_star[k]*(beta_plus[k] + beta_minus[k]) for k in range(P_CG)]) <= M, name="w_star_0")




    #---POSSIBLE WARM START (only for Gurobi on full model)
        if(len(warm_start) > 0):

            for i in range(P_CG):
                beta_plus[i].start  = max( warm_start[idx_CG[i]], 0)
                beta_minus[i].start = max(-warm_start[idx_CG[i]], 0)

    
        
#---IF PREVIOUS MODEL JUST UPDATE THE PENALIZATION
    else:
        #L0_Slope_CG = model.copy()
        xi          = [L0_Slope_CG.getVarByName('loss_'+str(i)) for i in range(N)]
        beta_plus   = [L0_Slope_CG.getVarByName('beta_+_'+str(idx_CG[i])) for i in range(P_CG)]
        beta_minus  = [L0_Slope_CG.getVarByName('beta_-_'+str(idx_CG[i])) for i in range(P_CG)]

        L0_Slope_CG.setObjective(quicksum(xi), GRB.MINIMIZE)

    print(L0_Slope_CG.Params)
    return L0_Slope_CG




#-----------------------------------------------ADD CONSTRAINTS--------------------------------------------------

def add_constraints_L0_Slope(X, y, L0_Slope_CG, w_star, alpha, beta_plus, beta_minus, b0, idx_CG, aux):
# Add a column in a model with respective constraints
    P_CG = len(idx_CG)

    L0_Slope_CG.addConstr( quicksum([ w_star[k]*(beta_plus[k] + beta_minus[k]) for k in range(P_CG)]) <= alpha, name="w_star_"+str(aux))
    L0_Slope_CG.update()    

    return L0_Slope_CG




#-----------------------------------------------ADD VARIABLES--------------------------------------------------

def add_columns_L0_Slope(X, y, L0_Slope_CG, violated_columns, violated_lambdas, idx_samples, dual_slacks, all_w_star_cstrts):

    new_betas_plus  = []
    new_betas_minus = []

    for j in range(len(violated_columns)):
        violated_column     = violated_columns[j]
        col_plus, col_minus = Column(), Column()


        ### Hinge constraints
        for i in range(len(idx_samples)):
            X_row_col = X[idx_samples[i], violated_column] 
                
            if X_row_col != 0:
                col_plus.addTerms(  y[idx_samples[i]]*X_row_col, dual_slacks[i])
                col_minus.addTerms(-y[idx_samples[i]]*X_row_col, dual_slacks[i])


        ### Slope constraints
        for i in range(len(all_w_star_cstrts)):
            col_plus.addTerms( violated_lambdas[j], all_w_star_cstrts[i])
            col_minus.addTerms(violated_lambdas[j], all_w_star_cstrts[i])
            
            
        new_betas_plus.append( L0_Slope_CG.addVar(lb=0, column=col_plus,  name="beta_+_"+str(violated_column) ) )
        new_betas_minus.append(L0_Slope_CG.addVar(lb=0, column=col_minus, name="beta_-_"+str(violated_column) ) )

    L0_Slope_CG.update()
    return L0_Slope_CG, new_betas_plus, new_betas_minus





