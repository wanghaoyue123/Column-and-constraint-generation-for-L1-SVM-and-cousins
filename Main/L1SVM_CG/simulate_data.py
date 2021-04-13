import numpy as np
import random
import math




def simulate_data(type_Sigma, N, P, k0, rho, mu, seed_X):

####### INPUT #######

# type_Sigma : in {1,2} defines the type of the simulation
	# type 1: x_+ \sim N(u_+, Sigma),  x_- \sim N(u_-, Sigma),  mu_+ =(mu,0,...,0,mu,0,....,0,mu...), mu_- = -mu_+, Sigma_ij = rho^|i-j|
	# type 2: x_+ \sim N(u_+, Sigma),  x_- \sim N(u_-, Sigma),  mu_+ =(mu,mu...mu, 0,...0),           mu_- = -mu_+, Sigma_ij = rho
	
# N,P     : size of the design matrix
# k0      : number of relevant variables  
# rho     : coefficient of correlation for Sigma
# mu      : used to define mu_+ and mu_-
# seed_X  : for random simulation


####### OUTPUT #######

# X, y
# l2_X : 2 norm of the columns of X (which have been normalized)


### Define X,y
	np.random.seed(seed=seed_X)
	X  = np.zeros((N, P))
	y  = np.ones(N)
	
### Define mu_positive, mu_negative
	mu_positive = np.zeros(P)

	if type_Sigma==1:
		index = [(2*i+1)*P/(2*k0) for i in range(k0)]  #equi-spaced k0
		mu_positive[index] = mu*np.ones(k0)

	elif type_Sigma==2:
		mu_positive[:k0] = mu*np.ones(k0)

	mu_negative = -mu_positive

### Define Sigma	
	if type_Sigma==1:
		Sigma = np.zeros(shape=(P,P))
		for i in range(P): 
			for j in range(P): Sigma[i,j]=rho**(abs(i-j))
				
		L = np.linalg.cholesky(Sigma)
	

	
#################### CASE 1: Sigma_ij = rho^(i-j) ####################
	if type_Sigma==1:

	### First half
		u_plus    = np.random.normal(size=(P,int(N/2)))
		X[:int(N/2),:] = np.dot(L, u_plus).T + mu_positive

	### Second half
		u_minus   = np.random.normal(size=(P,int(N/2)))
		X[int(N/2):,:] = np.dot(L, u_minus).T + mu_negative
		y[int(N/2):]   = -np.ones(int(N/2))

	### Shuffle
		X, y = shuffle(X, y)



#################### CASE 2: Sigma_ij = rho ####################
	elif type_Sigma==2:
		
	### First half
		X0_plus   = np.random.normal(size=(int(N/2),1))
		X[:int(N/2),:] = np.concatenate([np.random.normal(loc=  1./float(math.sqrt(1-rho))*mu_positive[:k0], size=(int(N/2),k0)), np.random.normal(size=(int(N/2),P-k0))], axis=1)
		
		X[:int(N/2),:] = math.sqrt(rho)*X0_plus + math.sqrt(1-rho)*X[:int(N/2),:]

	### Second half
		X0_minus   = np.random.normal(size=(int(N/2),1))
		X[int(N/2):,:]  = np.concatenate([np.random.normal(loc= 1./float(math.sqrt(1-rho))*mu_negative[:k0], size=(int(N/2),k0)), np.random.normal(size=(int(N/2),P-k0))], axis=1)        

		X[int(N/2):,:] = math.sqrt(rho)*X0_minus + math.sqrt(1-rho)*X[int(N/2):,:]        
		y[int(N/2):]   = -np.ones(int(N/2))

	### Shuffle
		X, y = shuffle(X, y)


	
### Nomalize X
	l2_X = []
	for i in range(P):
		l2 = np.linalg.norm(X[:,i])
		l2_X.append(l2)        
		X[:,i] = X[:,i]/float(l2)


	print('\nDATA CREATED for N='+str(N)+', P='+str(P)+', k0='+str(k0)+' Rho='+str(rho)+', Sigma='+str(type_Sigma)+', Seed_X='+str(seed_X)+', mu='+str(mu))
	return X, l2_X, y




def shuffle(X, y):

####### INPUT #######
# X: array of size (N,P)
# y: list of size (N,)

####### OUTPUT #######
# X,y: similarly shuffled 

	N, P = X.shape
	aux = np.concatenate([X,np.array(y).reshape(N,1)], axis=1)
	np.random.shuffle(aux)

	X = [aux[i,:P] for i in range(N)]
	y = [aux[i,P:] for i in range(N)]

	X = np.array(X).reshape(N,P)
	y = np.array(y).reshape(N,)

	return X, y

