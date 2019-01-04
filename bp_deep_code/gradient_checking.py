import numpy as np
from deep_neural import DeepNeural
class GradientCheck(object):
	def dictionary_to_vector(self,parameters):
		Keys = []
		count = 0
		parameters_info = {}
		for key in parameters.keys():
			new_vector = np.reshape(parameters[key],(-1,1))
			parameters_info[key] = parameters[key].shape
			Keys = Keys + [key]*new_vector.shape[0]
			if count == 0:
				theta = new_vector
			else:
				theta = np.concatenate((theta,new_vector),axis=0)
			count = count + 1
		return theta,parameters_info,Keys
	def gradients_to_vector(self,gradients):
		count = 0
		for key in gradients.keys():
			new_vector = np.reshape(gradients[key],(-1,1))
			if count == 0:
				theta = new_vector
			else:
				theta = np.concatenate((theta,new_vector),axis=0)
			count = count + 1
		return theta

	def vector_to_dictionary(self,theta,parameters_info):
		parameters = {}
		cumsum = 0
		for key in parameters_info.keys():
			n_row,n_col = parameters_info[key]
			sum = n_row * n_col
			parameters[key] = theta[cumsum:cumsum+sum].reshape((n_row,n_col))
			cumsum += sum
		return parameters
	def gradient_check_n(self,parameters,X,Y,epsilon=1e-7):
		parameters_values,parameters_info,_ = self.dictionary_to_vector(parameters)
		AL, caches = DeepNeural().L_model_forward(X, parameters)
		grads = DeepNeural().L_model_backward(AL, Y, caches)
		print(grads)
		grad = self.gradients_to_vector(grads)
		num_parameters = parameters_values.shape[0]
		print(parameters_values.shape)
		print(grad.shape)
		J_plus = np.zeros((num_parameters,1))
		J_minus = np.zeros((num_parameters,1))
		gradapprox = np.zeros((num_parameters,1))
		#grad = DeepNeural().L_model_backward()
		for i  in range(num_parameters):
			thetaplus = np.copy(parameters_values)
			thetaplus[i][0] = thetaplus[i][0] + epsilon
			ALplus,_ = DeepNeural().L_model_forward(X,self.vector_to_dictionary(thetaplus,parameters_info))
			J_plus[i] = DeepNeural().compute_cost(ALplus,Y)

			thetaminus = np.copy(parameters_values)
			thetaminus[i][0] = thetaminus[i][0] - epsilon
			ALminus,_ = DeepNeural().L_model_forward(X,self.vector_to_dictionary(thetaminus,parameters_info))
			J_minus[i] = DeepNeural().compute_cost(ALminus,Y)
			gradapprox[i] = (J_plus[i] - J_minus[i]) / (2*epsilon)
		numerator = np.linalg.norm(gradapprox - grad)
		denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
		difference = numerator / denominator
		if difference > 1e-7:
			print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
		else:
			print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
		return difference
if __name__ == "__main__":
	def gradient_check_n_test_case():
		np.random.seed(1)
		x = np.random.randn(4, 3)
		y = np.array([1, 1, 0]).reshape(1,3)
		W1 = np.random.randn(5, 4)
		b1 = np.random.randn(5, 1)
		W2 = np.random.randn(3, 5)
		b2 = np.random.randn(3, 1)
		W3 = np.random.randn(1, 3)
		b3 = np.random.randn(1, 1)
		parameters = {"W1": W1,
		              "b1": b1,
		              "W2": W2,
		              "b2": b2,
		              "W3": W3,
		              "b3": b3}

		return x, y, parameters
	X, Y, parameters = gradient_check_n_test_case()
	test = GradientCheck()
	test.gradient_check_n(parameters,X,Y)














