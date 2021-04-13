import numpy as np


def soft_thresholding_slope(y_arr, alpha_arr):
#y_arr:     vector
#alpha_arr: non negative and non increasing sequence of penalizations

	sign  = np.sign(y_arr)
	y_abs = np.abs(y_arr)

	arg    = np.argsort(y_abs)
	result = soft_thresholding_slope_positive(y_abs[arg][::-1], alpha_arr)[::-1] #sorted

	arg_bis    = np.argsort(arg)
	beta_slope = result[arg_bis]*sign

	return beta_slope



def soft_thresholding_slope_positive(y_arr, alpha_arr):
#y_arr:     non negative and non increasing sequence of vectors
#alpha_arr: non negative and non increasing sequence of penalizations
	
	n      = len(y_arr)
	idx_i  = np.zeros(n)
	idx_j  = np.zeros(n)
	s      = np.zeros(n)
	w      = np.zeros(n)
	result = np.zeros(n)

	k = 0;
	for i in range(n):
		idx_i[k] = i
		idx_j[k] = i
		s[k]     = y_arr[i] - alpha_arr[i]
		w[k]     = max(0, s[k])
	     
		while k>0 and w[k-1] <= w[k]:
			k -= 1
			idx_j[k] = i
			s[k]    += s[k+1]
			w[k]     = max(0, s[k] / (i - idx_i[k] + 1))
		k += 1
		#print idx_i, idx_j, w
	
	for j in range(k):
		for i in range(int(idx_i[j]), int(idx_j[j]+1)):
			result[i] = w[j]

	return result

#y = np.array([0.1,3.,-4.,2.])
#alpha = np.array([.4,.3,.2,.1])
#print soft_thresholding_slope(y, alpha)
