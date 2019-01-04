import numpy as np
from activation_function import Activation_func


class DeepNeural(object):
	def initial_parameters_deep(self,layers_dims):
		np.random.seed(1)
		parameters = {}
		L = len(layers_dims)
		for l in range(1,L):
			parameters['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
			parameters['b'+str(l)] = np.zeros((layers_dims[l],1))
			assert (parameters['W'+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
			assert (parameters['b'+str(l)].shape == (layers_dims[l],1))
		return parameters
	def linear_forward(self,A,W,b):
		'''
		Z[l] = W[l]A[l-1] + b[l]
		'''
		Z = np.dot(W,A) + b
		assert (Z.shape == (W.shape[0],A.shape[1]))
		cache = (A,W,b)
		return Z,cache
	def linear_activation_forward(self,A_prev,W,b,activation):
		'''
		A[l] = g(Z[l]) = g(W[l]A[l-1] + b[l])
		linear_cache:(A_prev,W,b) (A[l-1],W[l],b[l])
		activation_cache:Z = np.dot(W,A_prev) + b
		'''
		if activation == 'sigmoid':
			Z,linear_cache = self.linear_forward(A_prev,W,b)
			A,activation_cache = Activation_func().sigmoid_forward(Z)
		elif activation == 'relu':
			Z,linear_cache = self.linear_forward(A_prev,W,b)
			A,activation_cache = Activation_func().relu_forward(Z)
		assert (A.shape == (W.shape[0],A_prev.shape[1]))
		cache = (linear_cache,activation_cache)
		return A,cache
	def L_model_forward(self,X,parameters):
		caches = []
		A = X
		L = len(parameters) // 2
		for l in range(1,L):
			A_prev = A
			A,cache = self.linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation='relu')
			caches.append(cache)
		AL,cache = self.linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation='sigmoid')
		caches.append(cache)
		assert (AL.shape == (1,X.shape[1]))
		return AL,caches
	def compute_cost(self,AL,Y):
		'''J = -1/mSUM(i2m) (y(i)log(A[L](i)) + (1-y(i))log(1-A[L](i))'''
		m = Y.shape[1]
		logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
		cost = 1. / m * np.sum(logprobs)
		#cost = (np.dot(Y,np.log(AL).T) + np.dot((1-Y),np.log(1-AL).T)) / (-m)
		cost = np.squeeze(cost)
		assert (cost.shape == ())
		return cost
	def linear_backward(self,dZ,cache):
		'''
		dZ[L] = dJ/dA[L]*dA[L]/dZ[L] = A[L] - Y
		dW[L] = dJ/dA[L]*dA[L]/dZ[L]*dZ[L]/dW[L]
		      = dZ[L]*dZ[L]/dW[L] = 1/m*dZ[L]*A[L-1].T
		db[L] = dJ/dA[L]*dA[L]/dZ[L]*dZ[L]/db[L]
		      = dZ[L]*dZ[L]/db[L] = dZ[L]*I
		dA[L-1] = dJ/dA[L]*dA[L]/dZ[L]*dZ[L]/dA[L-1]
		        = dZ[L]*dZ[L]/dA[L-1] = W[L].T*dZ[L]
		'''
		A_prev,W,b = cache
		m = A_prev.shape[1]
		dW = np.dot(dZ,A_prev.T) / m
		db = np.sum(dZ,axis=1,keepdims=True) / m
		dA_prev = np.dot(W.T,dZ)
		assert (dA_prev.shape == A_prev.shape)
		assert (dW.shape == W.shape)
		assert (db.shape == b.shape)
		return dA_prev,dW,db
	def linear_activation_backward(self,dA,cache,activation):
		linear_cache,activation_cache = cache
		if activation == 'relu':
			dZ = Activation_func().relu_backward(dA,activation_cache)
			dA_prev,dW,db = self.linear_backward(dZ,linear_cache)
		elif activation == 'sigmoid':
			dZ = Activation_func().sigmoid_backward(dA,activation_cache)
			dA_prev,dW,db = self.linear_backward(dZ,linear_cache)
		return dA_prev,dW,db
	def L_model_backward(self,AL,Y,caches):
		grads = {}
		L = len(caches)
		m = AL.shape[1]
		Y = Y.reshape(AL.shape)
		dAL = -(np.divide(Y,AL) - np.divide((1-Y),(1-AL)))
		current_cache = caches[L-1]
		grads['dA'+str(L)],grads['dW'+str(L)],grads['db'+str(L)] = self.linear_activation_backward(dAL,current_cache,activation='sigmoid')
		for l in reversed(range(L-1)):
			current_cache = caches[l]
			dA_prev_temp,dW_temp,db_temp = self.linear_activation_backward(grads['dA'+str(l+2)],current_cache,activation='relu')
			grads['dA'+str(l+1)] = dA_prev_temp
			grads['dW'+str(l+1)] = dW_temp
			grads['db'+str(l+1)] = db_temp
		return grads
