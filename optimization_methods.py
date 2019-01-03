import numpy as np
import math
class Optimization(object):
	def random_mini_batches(self,X,Y,mini_batch_size=64,seed=0):
		'''
		first_mini_batch_X = shuffled_X[:,0:mini_batch_size]
		second_mini_batch_X = shuffled_X[:,mini_batch_size:2*mini_batc_size]
		'''
		np.random.seed(seed)
		m = X.shape[1] # number of training examples
		mini_batches = []
		# Step 1:Shuffle (X,Y)
		permutation = list(np.random.permutation(m))
		shuffled_X = X[:,permutation]
		shuffled_Y = Y[:,permutation].reshape((1,m))
		# Step 2: Partition (shuffled_X,shuffled_Y). Minus the end case
		num_complete_minibatches = math.floor(m/mini_batch_size)
		for k in range(0,num_complete_minibatches):
			mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
			mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
			mini_batch = (mini_batch_X,mini_batch_Y)
			mini_batches.append(mini_batch)
		if m % mini_batch_size != 0:
			mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:]
			mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:]
			mini_batch = (mini_batch_X,mini_batch_Y)
			mini_batches.append(mini_batch)
		return mini_batches
	def update_parameters_with_gd(self,parameters,grads,learnig_rate):
		L = len(parameters) // 2
		for l in range(L):
			parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learnig_rate*grads['dW'+str(l+1)]
			parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learnig_rate*grads['db'+str(l+1)]
		return parameters
	def initialize_velocity(self,parameters):
		L = len(parameters) // 2  # number of layers in the neural networks
		v = {}
		for l in range(L):
			v['dW'+str(l+1)] = np.zeros_like(parameters['W'+str(l+1)])
			v['db'+str(l+1)] = np.zeros_like(parameters['b'+str(l+1)])
		return v
	def update_parameters_with_momentum(self,parameters,grads,beta,learning_rate):
		'''
		Vdw[l] = beta*Vdw[l] + (1 - beta)dW[l]
		W[l] = W[l] - alpha*Vdw[l]
		Vdb[l] = beta*Vdb[l] + (1 - beta)db[l]
		b[l] = b[l] - alpha*Vdb[l]
		'''
		L = len(parameters) // 2
		v = self.initialize_velocity(parameters)
		for l in range(L):
			v['dW'+str(l+1)] = beta*v['dW'+str(l+1)] + (1-beta)*grads['dW'+str(l+1)]
			parameters['W'+str(l+1)] -= learning_rate*v['dW'+str(l+1)]
			v['db'+str(l+1)] = beta*v['db'+str(l+1)] + (1-beta)*grads['db'+str(l+1)]
			parameters['b'+str(l+1)] -= learning_rate*v['db'+str(l+1)]
		return parameters,v
	def initialize_adam(self,parameters):
		L = len(parameters) // 2
		v = {}
		s = {}
		for l in range(L):
			v['dW'+str(l+1)] = np.zeros_like(parameters['W'+str(l+1)])
			v['db'+str(l+1)] = np.zeros_like(parameters['b'+str(l+1)])
			s['dW'+str(l+1)] = np.zeros_like(parameters['W'+str(l+1)])
			s['db'+str(l+1)] = np.zeros_like(parameters['b'+str(l+1)])
		return v,s
	def update_parameters_with_adam(self,parameters,grads,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
		'''
		Vdw[l] = beta1*Vdw[l] + (1-beta1)*dJ/dW[l]
		VcorrecteddW[l] = Vdw[l]/(1-np.power(beta1,t)
		Sdw[l] = beta2*Sdw[l] + (1-beta2)*np.power(dJ/dW[l],2)
		ScorrecteddW[l] = SdW[l]/(1-np.power(beta2,t))
		W[l] = W[l] - alpha*VcorrecteddW[l]/np.sqr(ScorrecteddW[l] + epsilon)
		'''
		L = len(parameters) // 2
		v_corrected = {}
		s_corrected = {}
		v,s = self.initialize_adam(parameters)
		for l in range(L):
			v['dW'+str(l+1)] = beta1*v['dW'+str(l+1)] + (1-beta1)*grads['dW'+str(l+1)]
			v['db'+str(l+1)] = beta1*v['db'+str(l+1)] + (1-beta1)*grads['db'+str(l+1)]
			v_corrected['dW'+str(l+1)] = v['dW'+str(l+1)] / (1-np.power(beta1,t))
			v_corrected['db'+str(l+1)] = v['db'+str(l+1)] / (1-np.power(beta1,t))
			s['dW'+str(l+1)] = beta2*s['dW'+str(l+1)] + (1-beta2)*np.power(grads['dW'+str(l+1)],2)
			s['db'+str(l+1)] = beta2*s['db'+str(l+1)] + (1-beta2)*np.power(grads['db'+str(l+1)],2)
			s_corrected['dW'+str(l+1)] = s['dW'+str(l+1)] / (1-np.power(beta2,t))
			s_corrected['db'+str(l+1)] = s['db'+str(l+1)] / (1-np.power(beta2,t))
			parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*v_corrected['dW'+str(l+1)] / np.sqrt(s['dW'+str(l+1)] + epsilon)
			parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*v_corrected['db'+str(l+1)] / np.sqrt(s['db'+str(l+1)] + epsilon)
		return parameters,v,s









