import numpy as np
import random
from scipy.stats import norm
import math

import sys
# sys.path.append('../algorithms')
from heuristics_classification import *


def shuffle(X, y):

#X: array of size (N,P)
#y: list of size (N,)

    N, P = X.shape
    aux = np.concatenate([X,np.array(y).reshape(N,1)], axis=1)
    np.random.shuffle(aux)

    X = [aux[i,:P] for i in range(N)]
    y = [aux[i,P:] for i in range(N)]

    X = np.array(X).reshape(N,P)
    y = np.array(y).reshape(N,)

    return X,y








def simulate_data_group(type_Sigma, N, P, group_to_feat, k0, rho, tau_SNR, seed_X, f):


#GROUP_TO_FEAT : features associated to each group
#k0            : number of groups 

#RHO : correlation_coefficient
#TAU : controle entre mu + et - OR SNR
#SEED X : for random simulation

#SIGMA_1 = rho^(i-j), sgn( N(0, Sigma_group)*mu_+ + epsilon)
#SIGMA_2 = rho,       sgn( N(0, Sigma_group)*mu_+ + epsilon)



    np.random.seed(seed=seed_X)
    

#------------BETA-------------
    u_positive = np.zeros(P)

    for idx in range(k0):
        u_positive[group_to_feat[idx]] = tau_SNR*np.ones(len(group_to_feat[idx]))
    u_negative = -u_positive



#------------X_train and X_test-------------
    X_train = np.zeros(shape=(N,P))
    y_train = []
    
    X_test=np.zeros(shape=(N,P))
    y_test = []
    




#---CASE 2: RHO 
    if(type_Sigma==2):

        
    #------------X_train-------------
        X_plus  = np.zeros((int(N/2), 0))
        X_minus = np.zeros((int(N/2), 0))

        G = len(group_to_feat)
        P = np.sum([len(group_to_feat[idx]) for idx in range(G)])

    #---Simulate once
        X0_plus  = np.random.normal(size=(int(N/2), G))
        Xi_plus  = np.random.normal(loc=  1./float(math.sqrt(1-rho))*u_positive, size=(int(N/2), P))
        
        X0_minus = np.random.normal(size=(int(N/2), G))
        Xi_minus = np.random.normal(loc=  1./float(math.sqrt(1-rho))*u_negative, size=(int(N/2), P))

        for idx in range(G): 
            sub_X0_plus = np.array([X0_plus[:, idx] for _ in group_to_feat[idx]]).T
            sub_Xi_plus = Xi_plus[:, group_to_feat[idx]]
            X_plus      = np.concatenate([X_plus, math.sqrt(rho)*sub_X0_plus + math.sqrt(1-rho)*sub_Xi_plus ], axis=1)

            sub_X0_minus = np.array([X0_minus[:, idx] for _ in group_to_feat[idx]]).T
            sub_Xi_minus = Xi_minus[:, group_to_feat[idx]]
            X_minus      = np.concatenate([X_minus, math.sqrt(rho)*sub_X0_minus + math.sqrt(1-rho)*sub_Xi_minus ], axis=1)


    #---Concatenate
        X_train = np.concatenate([X_plus, X_minus])
        y_train = np.concatenate([np.ones(int(N/2)), -np.ones(int(N/2))])
        X_train, y_train = shuffle(X_train, y_train)

        print(X_train.shape)







#------------NORMALIZE------------- 
    
#---Normalize all the X columns
    l2_X_train = []

    for i in range(P):
        l2 = np.linalg.norm(X_train[:,i])
        l2_X_train.append(l2)        
        X_train[:,i] = X_train[:,i]/float(l2)


    write_and_print('\nDATA CREATED for N='+str(N)+', P='+str(P)+', k0='+str(k0)+' Rho='+str(rho)+' Sigma='+str(type_Sigma)+' Seed_X='+str(seed_X), f)

    return X_train, l2_X_train, y_train, u_positive


