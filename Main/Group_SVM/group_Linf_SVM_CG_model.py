import numpy as np
from gurobipy import *

# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


#-----------------------------------------------BUILD THE MODEL--------------------------------------------------

def group_Linf_SVM_CG_model(X, y, group_to_feat, idx_groups_CG, alpha, time_limit, group_Linf_SVM_CG, warm_start, f, duality_gap=0):

#group_to_feat    = for each index of a group we have access to the list of features belonging to this group
#idx_groups_CG    = index of the groups used to generate 
#group_Linf_SVM_CG      = previous model to speed up
#WARM START       = speed up Gurobi on full model 
    

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P      = X.shape
    n_groups = len(idx_groups_CG)

    write_and_print('Number of constraints: '+str(N), f)
    write_and_print('Number of groups     : '+str(n_groups), f)

    
    if group_Linf_SVM_CG == 0:
    
    #---VARIABLES
        group_Linf_SVM_CG=Model("group_Linf_SVM_CG")
        group_Linf_SVM_CG.setParam('OutputFlag', False )
        #group_Linf_SVM_CG.setParam('TimeLimit', time_limit)
        if duality_gap>0: group_Linf_SVM_CG.params.BarConvTol = duality_gap
        
        
        #Hinge loss
        xi = np.array([group_Linf_SVM_CG.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])
        max_groups = []        


        #Groups in the model
        for idx in idx_groups_CG:
            max_groups.append( group_Linf_SVM_CG.addVar(lb=0, name="max_group_"+str(idx)) )
            for j in group_to_feat[idx]:
                group_Linf_SVM_CG.addVar(lb=0, name="beta_+_"+str(j)) 
                group_Linf_SVM_CG.addVar(lb=0, name="beta_-_"+str(j)) 
        
        b0 = group_Linf_SVM_CG.addVar(lb=-GRB.INFINITY, name="b0")

        group_Linf_SVM_CG.update()


    #---OBJECTIVE VALUE 
        group_Linf_SVM_CG.setObjective(quicksum(xi) + alpha*quicksum(max_groups[i] for i in range(n_groups)), GRB.MINIMIZE)


    #---HIGE CONSTRAINTS
        for i in range(N):
            group_Linf_SVM_CG.addConstr(xi[i] + y[i]*(b0 + quicksum([ quicksum([ X[i][j]*(group_Linf_SVM_CG.getVarByName('beta_+_'+str(j)) - group_Linf_SVM_CG.getVarByName('beta_-_'+str(j)) )
                for j in group_to_feat[idx]])  for idx in idx_groups_CG]) )>= 1, name="slack_"+str(i))


    #---INFINITE NORM CONSTRAINTS
        for idx in idx_groups_CG:
            max_group  = group_Linf_SVM_CG.getVarByName('max_group_'+str(idx))

            for j in group_to_feat[idx]:    
                beta_plus  = group_Linf_SVM_CG.getVarByName('beta_+_'+str(j))
                beta_minus = group_Linf_SVM_CG.getVarByName('beta_-_'+str(j))
                group_Linf_SVM_CG.addConstr( max_group - beta_plus - beta_minus >= 0, name="group_"+str(idx)+"feat_"+str(j))


    #---POSSIBLE WARM START (only for Gurobi on full model)
        if(len(warm_start) > 0):
            print(0)


    
    
        
#---IF PREVIOUS MODEL JUST UPDATE THE OBJECTIVE
    else:
        xi          = [group_Linf_SVM_CG.getVarByName('loss_'+str(i)) for i in range(N)]
        max_groups  = [group_Linf_SVM_CG.getVarByName('max_group_'+str(idx)) for idx in idx_groups_CG]

        group_Linf_SVM_CG.setObjective(quicksum(xi) + alpha*quicksum(max_groups[i] for i in range(n_groups)), GRB.MINIMIZE)
   

    group_Linf_SVM_CG.update()
    
    return group_Linf_SVM_CG







#-----------------------------------------------ADD VARIABLES--------------------------------------------------
#-----------------------------------------------PROBLEM N+P CONTRAINTES ??--------------------------------------------------


def add_groups_Linf_SVM(X, y, group_to_feat, group_Linf_SVM_CG, idx_groups_to_add, idx_samples, cstrts_slacks, alpha):

#idx_samples:   samples in the model -> may be different from range(N) when deleting samples
#cstrts_slacks: constraints in the model
    
    for idx in idx_groups_to_add:

    #---NEW GROUP
        max_group = group_Linf_SVM_CG.addVar(lb=0, obj = alpha, name="max_group_"+str(idx))

        for j in group_to_feat[idx]:

        #---NEW FEATURES IN THE GROUP
            col_plus, col_minus = Column(), Column()

            for i in range(len(idx_samples)):
                col_plus.addTerms(  y[idx_samples[i]]*X[idx_samples[i]][j], cstrts_slacks[i]) #coeff for constraint, name of constraint
                col_minus.addTerms(-y[idx_samples[i]]*X[idx_samples[i]][j], cstrts_slacks[i])
                
                
            beta_plus  = group_Linf_SVM_CG.addVar(lb=0, obj = 0, column=col_plus,  name="beta_+_"+str(j) )
            beta_minus = group_Linf_SVM_CG.addVar(lb=0, obj = 0, column=col_minus, name="beta_-_"+str(j) )
            group_Linf_SVM_CG.update()


        #---INFINITE NORM CONSTRAINTS
            group_Linf_SVM_CG.addConstr( max_group - beta_plus - beta_minus >= 0, name="group_"+str(idx)+"feat_"+str(j))
            group_Linf_SVM_CG.update()
            

    return group_Linf_SVM_CG





