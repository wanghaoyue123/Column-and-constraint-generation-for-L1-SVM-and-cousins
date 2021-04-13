import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
import time
import math


from algorithm1_classification import *
from Gurobi_SVM import *




def heuristic(type_descent, type_loss, type_penalization, X, y, K0_list, N_alpha, X_add, mu_max, epsilon, f):
    
#Type_descent: indicates whihc type of heuristic :
#  - if UP we increase k
#  - if DOWN we decrease k


#TYPE_LOSS = 1 : HINGE LOSS 
#TYPE_LOSS = 2 : SQUARED HINGE LOSS

#TYPE_PENALIZATION = 1 : L1 
#TYPE_PENALIZATION = 2 : L2
	


    write_and_print('\n\nHEURISTIC '+str(type_descent)+' for '+str(type_loss)+' loss and '+str(type_penalization)+ ' penalization: ',f)
    
    N,P   = X.shape
    start = time.time()


#---RESULTS
    beta_list         = [[[] for i in range(N_alpha)] for K0 in K0_list]
    train_errors_list = [[[] for i in range(N_alpha)] for K0 in K0_list]
    accross_alpha     = [0  for K0 in K0_list]
    accross_k         = [0  for K0 in K0_list]
    betas_l1_l2_SVM   = [] #useful for later comparison
    errors_l1_l2_SVM  = []

    
       
#---ALPHA LIST
    dict_alpha_max = {'l1': np.max(np.sum( np.abs(X), axis=0)), 'l2': 2*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)])}
    alpha_max      = dict_alpha_max[type_penalization]
    alpha_list     = [alpha_max*0.8**i for i in range(N_alpha)]
    

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   

#---INITIALIZATION
    dict_score_max = {'squared_hinge': 1, 'hinge': 1, 'logreg': math.log(2)}

#---alpha_max
    for K0 in K0_list: 
        beta_list[K0][0]         = (np.zeros(P), 0)
        train_errors_list[K0][0] = N*dict_score_max[type_loss]
        

#---K0 = 0 
    for loop in range(1, N_alpha):
        beta_list[0][loop]         = (np.zeros(P), 0)
        train_errors_list[0][loop] = N*dict_score_max[type_loss]

    
#---K0_max if we decrease K0
    if type_descent   == 'down':
        K0_max       = K0_list[::-1][0]
        write_and_print('K= '+str(K0_max),f)

    #---Compute for decreasing values
        for loop in range(N_alpha):
            alpha = alpha_list[loop]

            if type_loss=='hinge' and type_penalization=='l1' : #keep the model at every iteration hence we don't use estimator_on_support
                beta_start = [] if loop == 0 else beta
                
                beta, beta_0, error, _ = Gurobi_SVM('hinge', 'l1', 'no_L0', X, y, alpha) 
            else: 
                beta, beta_0, error    = estimator_on_support(type_loss, type_penalization, X, y, alpha, np.ones(P))
                
            betas_l1_l2_SVM.append((beta, beta_0))
            errors_l1_l2_SVM.append(error)
            
            beta_K0_max, beta_0_K0_max, train_error = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, np.copy((beta, beta_0)), X_add, epsilon, highest_eig=mu_max)
            beta_list[K0_max][loop]                 = (beta_K0_max, beta_0_K0_max)
            train_errors_list[K0_max][loop]         = train_error

   
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

#---MAIN LOOP 
    if type_descent == 'down': K0_list = K0_list[::-1]

    for K0 in K0_list[1:]: 
        if K0 > 0:

            write_and_print('K= '+str(K0),f)
            for loop in range(1, N_alpha):
                alpha = alpha_list[loop]

                dict_up_down = {'up':-1, 'down':1}
                delta_k      = int(dict_up_down[type_descent])
                
                beta_1, beta0_1, train_error_1 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, np.copy(beta_list[K0][loop-1]),       X_add, epsilon, highest_eig=mu_max)
                beta_2, beta0_2, train_error_2 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, np.copy(beta_list[K0+delta_k][loop]), X_add, epsilon, highest_eig=mu_max)


                if train_error_1 < train_error_2:
                    accross_alpha[K0] += 1
                    beta_list[K0][loop]         = (beta_1, beta0_1)
                    train_errors_list[K0][loop] = train_error_1
                
                
                else:
                    accross_k[K0] += 1
                    beta_list[K0][loop]         = (beta_2, beta0_2)
                    train_errors_list[K0][loop] = train_error_2


    end_time = time.time()-start
                                                                   
    write_and_print('Moves across alpha : '+str(accross_alpha),f)
    write_and_print('Moves across K     : '+str(accross_k),f)
    write_and_print('Time               : '+str(round(end_time,2)),f)


    return beta_list, train_errors_list, alpha_list, betas_l1_l2_SVM, errors_l1_l2_SVM, end_time







def compute_L1_L2(type_loss, type_penalization, X, y, N_alpha, X_add, f):

#TYPE_LOSS = HINGE LOSS or QUARED HINGE LOSS
#TYPE_PENALIZATION = 1 : L1 or L2

    write_and_print('\n\BENCHMARK for '+str(type_loss)+' loss and '+str(type_penalization)+ ' penalization: ',f)
    N,P   = X.shape
    
    betas  = []
    errors = []
    start_time = time.time()
       

    dict_alpha_max = {'l1': np.max(np.sum( np.abs(X), axis=0)), 'l2': 2*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)])}
    alpha_max      = dict_alpha_max[type_penalization]
    alpha_list     = [alpha_max*0.8**i for i in range(N_alpha)] 


    for loop in range(N_alpha):
        alpha = alpha_list[loop]

        if type_loss=='hinge' and type_penalization=='l1' : #keep the model at every iteration hence we don't use estimator_on_support
            beta_start = [] if loop == 0 else beta
            beta, beta_0, error, _ = Gurobi_SVM('hinge', 'l1', 'no_L0', X, y, alpha) 
        else: 
            beta, beta_0, error   = estimator_on_support(type_loss, type_penalization, X, y, alpha, np.ones(P))
            
        betas.append((beta, beta_0))
        errors.append(error)

    end_time = time.time()-start_time
    return betas, errors, end_time








def best_of_up_down(train_errors_up, train_errors_down, betas_up, betas_down, K0_list, N_alpha, f):

#Keep the best betas between up and down, and the best train errors

    write_and_print('\nBEST OF UP / DOWN ?', f)

#---Results
    best_betas        = [[] for K0 in K0_list]
    best_train_errors = [[] for K0 in K0_list]
    up, down          = [],[]

#---Loop
    for K0 in K0_list:
        c,d=0,0

        for loop in range(N_alpha):
            a = train_errors_up[K0][loop]
            b = train_errors_down[K0][loop]

            if(a<b):
                c+=1
                best_betas[K0].append(np.copy(betas_up[K0][loop]))
                best_train_errors[K0].append(a)

            else:
                d+=1
                best_betas[K0].append(np.copy(betas_down[K0][loop]))
                best_train_errors[K0].append(b)

        up.append(c)
        down.append(d)
     
    write_and_print('UP    : '+str(up), f)
    write_and_print('DOwN  : '+str(down), f)

    return best_betas, best_train_errors



def write_and_print(text,f):
    print(text)
    f.write('\n'+text)


