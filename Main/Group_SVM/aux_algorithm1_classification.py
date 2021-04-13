import math
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from Gurobi_SVM import *



def estimator_on_support(type_loss, type_penalization, X, y, alpha, beta):

#TYPE LOSS        : one of the parameters squared_hinge', 'hinge', 'logreg'
#TYPE PENALIZATION: one of the parameters l1', 'l2'
    
	N,P    = X.shape
	ones_N = np.ones(N)

#---Support not empty -> carefull intercept term
	support = np.where(beta[:P]!=0)[0]

	if len(support) > 0:

		X_support = np.array([X[:,support[i]] for i in range(len(support))]).T

		if type_loss=='hinge' and type_penalization=='l1' :
			#---Gurobi without model or warm-start
			time_limit = 30
			#beta, beta_0, error, _ = Gurobi_SVM('hinge', 'l1', X, y, support, alpha, 0, beta)
			beta_support, beta_0, error, _ = Gurobi_SVM('hinge', 'l1', 'no_L0', X_support, y, alpha)
			beta          = np.zeros(P)
			beta[support] = beta_support



		else:
			dict_loss  = {'squared_hinge': True, 'hinge': True, 'logreg': False}
			dict_pen   = {'l1':(False, 1.), 'l2':(dict_loss[type_loss], 0.5)}
			aux_loss  = dict_pen[type_penalization]

			if type_loss == 'logreg':
				estimator = LogisticRegression(penalty=type_penalization, dual=aux_loss[0], C=aux_loss[1]/alpha) #http://scikit-learn.org/stable/modules/linear_model.html#linear-model
			else:
				estimator = LinearSVC(penalty=type_penalization, loss= type_loss, dual=aux_loss[0], C=aux_loss[1]/alpha)

		#---Define beta 
			estimator.fit(X_support, y)
			beta_support  = estimator.coef_[0]
			beta_0        = estimator.intercept_[0] 
			beta          = np.zeros(P)
			beta[support] = beta_support
       
    
		#---Compute error
			aux       = y*(np.dot(X_support, beta_support)+ beta_0*ones_N)
			dict_loss = {'squared_hinge': [max(0, 1 - aux[i])**2           for i in range(N)],
						 'hinge'        : [max(0, 1 - aux[i])              for i in range(N)],
						 'logreg'       : [math.log(1 + math.exp(-aux[i])) for i in range(N)]
						}

			dict_penalization = {'l1': np.sum(np.abs(beta_support)), 'l2': np.linalg.norm(beta_support)**2}

			error = np.sum(dict_loss[type_loss]) + alpha*dict_penalization[type_penalization]
	    
	    
#---Support empty
	else:
		dict_score_max = {'squared_hinge': 1, 'hinge': 1, 'logreg': math.log(2)}
		beta, beta_0   = beta[:P], beta[P] 
		error  = N*dict_score_max[type_loss]
        
	return beta, beta_0, error



############## SOFT THRESHOLDING OPERATORS ################

def soft_thresholding_l0(c,alpha):
    return c


def soft_thresholding_l1(c,alpha):
    if(alpha>=abs(c)):
        return 0
    else:
        if (c>=0):
            return c-alpha
        else:
            return c+alpha

    
def soft_thresholding_l2(c,alpha):
    return c/float(1+2*alpha)



