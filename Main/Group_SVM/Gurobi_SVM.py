import numpy as np
from gurobipy import *
import datetime

import sys
#sys.path.append('/cm/shared/engaging/gurobi/gurobi701/linux64/lib/python2.7')




def Gurobi_SVM(type_loss, type_penalization, is_L0, X, y, alpha, K=0, beta_start=[], time_limit=300, Big_M=0, OutputFlag=False, Gurobi_SVM=0):

## Gurobi_SVM: previous model

    N,P  = X.shape
    


##################################### USUALLY NO PREVIOUS MODEL #####################################

    if Gurobi_SVM == 0:
    

    #---VARIABLES
        Gurobi_SVM=Model("Gurobi_SVM")
        Gurobi_SVM.setParam('OutputFlag', OutputFlag )
        Gurobi_SVM.setParam('TimeLimit', time_limit)
        
        
        #Hinge loss
        xi = np.array([Gurobi_SVM.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])

        beta_plus  = np.array([Gurobi_SVM.addVar(lb=0, name="beta_+_"+str(i)) for i in range(P)])
        beta_minus = np.array([Gurobi_SVM.addVar(lb=0, name="beta_-_"+str(i)) for i in range(P)])
        b0 = Gurobi_SVM.addVar(lb=-GRB.INFINITY, name="b0")

        #L0 control
        if is_L0 == 'L0':
            z = [Gurobi_SVM.addVar(0.0, 1.0, 1.0, GRB.BINARY, name="z_"+str(i)) for i in range(P)]

        Gurobi_SVM.update()



    #---OBJECTIVE VALUE WITH HINGE LOSS AND L1-NORM
        dict_loss = {'hinge': quicksum(xi),
                     'squared_hinge': quicksum(xi[i]*xi[i] for i in range(N))}

        dict_penalization = {'l1': quicksum(  beta_plus[i]+beta_minus[i]                               for i in range(P)),
                             'l2': quicksum( (beta_plus[i]+beta_minus[i])*(beta_plus[i]+beta_minus[i]) for i in range(P))}

        Gurobi_SVM.setObjective(dict_loss[type_loss] + alpha*dict_penalization[type_penalization], GRB.MINIMIZE)


    #---HIGE CONSTRAINTS
        for i in range(N):
            Gurobi_SVM.addConstr(xi[i] + y[i]*(b0 + quicksum([ X[i][j]*(beta_plus[j] - beta_minus[j]) for j in range(P)]))>= 1, 
                                 name="slack_"+str(i))



    #---SPARSITY
        if is_L0 == 'L0':
        #---Big M constraint
            Big_M = 2*np.max(np.abs(beta_start))

            for i in range(P):
                Gurobi_SVM.addConstr( beta_plus[i] + beta_minus[i]   <= Big_M*z[i], "max_beta_"+str(i))
                Gurobi_SVM.addConstr(-beta_plus[i] - beta_minus[i]   <= Big_M*z[i], "min_beta_"+str(i))

        #---Sparsity constraint
            Gurobi_SVM.addConstr(quicksum(z) <= K, "sparsity")




    #---POSSIBLE WARM START when computing L1 -> can be imporved in keeping model...
        if len(beta_start) > 0:

            for i in range(P):
                beta_plus[i].start  = max( beta_start[i], 0)
                beta_minus[i].start = max(-beta_start[i], 0)




##################################### SOMETIMES PREVIOUS MODEL #####################################

    else:
        #L1_SVM_CG = model.copy()
        xi          = [Gurobi_SVM.getVarByName('loss_'+str(i))   for i in range(N)]
        beta_plus   = [Gurobi_SVM.getVarByName('beta_+_'+str(i)) for i in range(P)]
        beta_minus  = [Gurobi_SVM.getVarByName('beta_-_'+str(i)) for i in range(P)]

        dict_loss = {'hinge': quicksum(xi),
                     'squared_hinge': quicksum(xi[i]*xi[i] for i in range(N))}

        dict_penalization = {'l1': quicksum(  beta_plus[i]+beta_minus[i]                               for i in range(P)),
                             'l2': quicksum( (beta_plus[i]+beta_minus[i])*(beta_plus[i]+beta_minus[i]) for i in range(P))}

        Gurobi_SVM.setObjective(dict_loss[type_loss] + alpha*dict_penalization[type_penalization], GRB.MINIMIZE)





###########################################################################################################
        
#---RESULTS
    Gurobi_SVM.update()
    Gurobi_SVM.optimize()
    
    beta    = np.array([beta_plus[j].X - beta_minus[j].X for j in range(P)])
    beta_0  = Gurobi_SVM.getVarByName('b0').X 
    obj_val = Gurobi_SVM.ObjVal  

    return beta, beta_0, obj_val, Gurobi_SVM


