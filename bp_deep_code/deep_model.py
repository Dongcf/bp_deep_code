import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
import sklearn.datasets
import json
import h5py
from optimization_methods import Optimization
from deep_neural import DeepNeural
class L_layers_model(object):
	def __init__(self,X,Y,layers_dims,optimizer,learning_rate=0.0075,mini_batch_size=64,beta=0.9,beta1=0.9,beta2=0.999,epsilon=1e-8,num_epochs=10000,print_cost=True):
		self.X = X
		self.Y = Y
		self.layers_dims = layers_dims
		self.optimizer = optimizer
		self.learning_rate = learning_rate
		self.mini_batch_size = mini_batch_size
		self.beta = beta
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.num_epochs = num_epochs
		self.print_cost = print_cost
	def model(self):
		np.random.seed(1)
		costs = []
		t = 0
		seed = 10
		parameters = DeepNeural().initial_parameters_deep(self.layers_dims)
		for i in range(self.num_epochs):
			seed += 1
			minibatches = Optimization().random_mini_batches(self.X,self.Y,self.mini_batch_size,seed)
			for minibatche in minibatches:
				(minibatche_X,minibatche_Y) = minibatche
				AL,caches = DeepNeural().L_model_forward(minibatche_X,parameters)
				cost = DeepNeural().compute_cost(AL,minibatche_Y)
				grads = DeepNeural().L_model_backward(AL,minibatche_Y,caches)
				if self.optimizer == 'gd':
					parameters = Optimization().update_parameters_with_gd(parameters,grads,self.learning_rate)
				if self.optimizer == 'momentum':
					parameters,v = Optimization().update_parameters_with_momentum(parameters,grads,0.9,self.learning_rate)
				if self.optimizer == 'adam':
					t += 1
					parameters,v,s= Optimization().update_parameters_with_adam(parameters,grads,t,self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8)
			if self.print_cost and i % 100 == 0:
				print("Cost after iteration %i:%f" %(i,cost))
			if self.print_cost and i % 10 == 0:
				costs.append(cost)
		plt.plot(np.squeeze(costs))
		plt.ylabel('costs')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(self.learning_rate))
		plt.show()
		f = open("parameters.txt",'w')
		f.write(str(parameters))
		f.close()
		return parameters
	def predict(self,X,y):
		m = X.shape[1]
		with open("parameters.txt",'r') as f:
			parameters = json.load(f)
		n = len(parameters) // 2
		p = np.zeros((1,m))
		probas,caches = DeepNeural.L_model_forward(X,parameters)
		for i in range(0,probas.shape[1]):
			if probas[0,i] > 0.5:
				p[0,i] = 1
			else:
				p[0,i] = 0
		print("Accuracy:" + str(np.sum((p == y)/m)))
		return p
if __name__ =="__main__":
	def load_data(train_data, test_data):
		train_dataset = h5py.File(train_data, 'r')
		train_set_x_orig = np.array(train_dataset['train_set_x'][:])
		train_set_y_orig = np.array(train_dataset['train_set_y'][:])
		test_dataset = h5py.File(test_data, 'r')
		test_set_x_orig = np.array(test_dataset['test_set_x'][:])
		test_set_y_orig = np.array(test_dataset['test_set_y'][:])
		classes = np.array(test_dataset['list_classes'][:])
		train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
		test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
		return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
	test_data = "datasets\\test_catvnoncat.h5"
	train_data = "datasets\\train_catvnoncat.h5"
	train_x_orig, train_y, test_x_orig, test_y, classes = load_data(train_data, test_data)
	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
	train_x = train_x_flatten / 255.
	test_x = test_x_flatten / 255.
	layers_dims = [12288, 20, 7, 5, 1]
	task1 = L_layers_model(train_x,train_y,layers_dims,"gd")
	task1.model()















