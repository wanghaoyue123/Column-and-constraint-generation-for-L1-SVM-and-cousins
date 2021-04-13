import numpy as np
from gurobipy import *
from L1_SVM_CP_model import *

import time
# sys.path.append('../synthetic_datasets')
from simulate_data_classification import *



def L1_SVM_CP(X_train, y_train, index_samples, alpha, epsilon_RC, time_limit, model, warm_start, f):


#INPUT
#n_features_RFE: number of features to give to RFE to intialize
#epsilon_RC    : maximum non negatuve reduced cost

    N,P       = X_train.shape
    N_CP      = len(index_samples)
    ones_N    = np.ones(N)
    is_L1_SVM = N_CP == N

    constraint_to_check = list(set(range(N))-set(index_samples))

    constraint_original = list(index_samples)
    constraint_added = []


#---Build the model
    start = time.time()
    model = L1_SVM_CP_model(X_train, y_train, index_samples, alpha, time_limit, model, warm_start, f) #model=0 -> no warm start
    
#---Infinite loop until all the constraints are satisfied
    continue_loop = True
    aux = 0

    while continue_loop: 
        continue_loop = False
        aux += 1

    #---Solve the relax problem to get the dual solution
        model.optimize()
        write_and_print('Time optimizing = '+str(time.time()-start), f)



        
        

        if N_CP != N:
        #---Parameters
            betas_plus  = [model.getVarByName('beta_+_'+str(idx)) for idx in range(P)]
            betas_minus = [model.getVarByName('beta_-_'+str(idx)) for idx in range(P)]
            beta        = np.array([beta_plus.X  for beta_plus  in betas_plus]) - np.array([beta_minus.X for beta_minus in betas_minus])
            b0         = model.getVarByName('b0')
            b0_val     = b0.X
            

        #---Look for constraints with negative reduced costs
            RC_aux           = np.dot(X_train[np.array(constraint_to_check), :], beta) + b0_val*ones_N[:N-N_CP]
            RC_array         = ones_N[:N-N_CP] - y_train[np.array(constraint_to_check)]*RC_aux

            violated_constraints = np.array(constraint_to_check)[RC_array > epsilon_RC]
            #violated_constraints = np.array(constraint_to_check)




        #---Add the constraints with negative reduced costs
            n_constraints_to_add = violated_constraints.shape[0]
            N_CP += n_constraints_to_add

            if n_constraints_to_add>0:
                continue_loop = True
                
                write_and_print('Number of constraints added: '+str(n_constraints_to_add), f)
                write_and_print('Max violated constraint    : '+str(round(np.max(RC_array), 3)), f)

                model = add_constraints_L1_SVM(X_train, y_train, model, violated_constraints, betas_plus, betas_minus, b0, range(P)) 

                for violated_constraint in violated_constraints:
                    index_samples.append(violated_constraint)
                    constraint_to_check.remove(violated_constraint)
                    constraint_added.append(violated_constraint)



#---Solution
    write_and_print('Number of rounds: '+str(aux), f)

    betas_plus  = [model.getVarByName('beta_+_'+str(idx)) for idx in range(P)]
    betas_minus = [model.getVarByName('beta_-_'+str(idx)) for idx in range(P)]
    beta        = np.array([beta_plus.X  for beta_plus  in betas_plus]) - np.array([beta_minus.X for beta_minus in betas_minus])
    b0          = model.getVarByName('b0').X 
    
    obj_val = model.ObjVal

#---TIME STOPS HERE
    time_CP = time.time()-start 
    write_and_print('\nTIME CP = '+str(time_CP), f)


#---support and Objective value
    support = np.where(beta!=0)[0]
    write_and_print('\nObj value   = '+str(obj_val), f)
    write_and_print('Len support = '+str(len(support)), f)





#---Violated constraints and dual support
    constraints = np.ones(N_CP) - y_train[np.array(index_samples)]*( np.dot(X_train[np.array(index_samples), :], beta) + b0*np.ones(N_CP))
    violated_constraints = np.arange(N_CP)[constraints >= 0]
    write_and_print('\nNumber violated constraints =  '+str(violated_constraints.shape[0]), f)

    solution_dual = np.array([model.getConstrByName('slack_'+str(idx)).Pi for idx in index_samples])
    support_dual  = np.where(solution_dual!=0)[0]
    write_and_print('Len support dual = '+str(len(support_dual)), f)

    write_and_print('Solution dual = '+str(np.sum(solution_dual)), f)
    write_and_print('Solution primal test = '+str(np.sum( [max(0,con) for con in constraints] )+alpha*np.sum(np.abs(beta)) ) , f)




    #solution_dual = np.array([model.getConstrByName('slack_'+str(idx)).Pi for idx in range(N)])
    #print np.min(solution_dual), np.max(solution_dual), np.dot(y_train, solution_dual)
    #aux = np.dot(X_train.T, y_train*solution_dual)
    #print alpha - np.max(np.abs(aux))

    ##constraints = np.ones(N_CP) - y_train*( np.dot(X_train, beta) + b0*np.ones(N_CP))
    #xi = np.array([model.getVarByName('loss_'+str(i)).X  for i in index_samples])
    #print np.sum(xi), np.sum([max(0,con) for con in constraints])
    #for i in constraint_original:
    #    if round(xi[i] - max(constraints[i], 0), 4) != 0:
    #        print xi[i], max(constraints[i], 0)


    return beta, b0, support, time_CP, model, index_samples, obj_val









