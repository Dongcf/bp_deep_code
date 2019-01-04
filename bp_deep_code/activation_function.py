import numpy as np
class Activation_func(object):
	def sigmoid_forward(self,Z):
		A = 1. / (1. + np.exp(-Z))
		cache = Z
		return A,cache
	def sigmoid_backward(self,dA,cache):
		'''
		dZ[l] = dJ/dA[l] * dA[l]/dZ[l]
		dsigmoid(Z) = sigmoid(Z)*[1-sigmoid(Z)]
		'''
		Z = cache
		s = 1. / (1. + np.exp(-Z))
		dZ = dA * s * (1 - s)
		return dZ
	def relu_forward(self,Z):
		A = np.maximum(0,Z)
		assert (A.shape == Z.shape)
		cache = Z
		return A,cache
	def relu_backward(self,dA,cache):
		'''
		dZ[l] = dJ/dA[l] * dA[l]/dZ[l]
		drelu(Z) = 1 if Z>0 else 0
		'''
		Z = cache
		dZ = np.array(dA,copy=True)
		dZ[Z <= 0] = 0
		assert (dZ.shape == Z.shape)
		return dZ
	def tanh_forward(self,Z):
		A = np.tanh(Z)
		cache = Z
		return A,cache
	def tanh_backward(self,dA,cache):
		'''
		dZ[l] = dJ/dA[l] * dA[l]/dZ[l]
		dtanh(Z) = 1 - tanh(Z)**2
		'''
		Z = cache
		tanh = np.tanh(Z)
		dZ = 1 - tanh**2
		return dZ





