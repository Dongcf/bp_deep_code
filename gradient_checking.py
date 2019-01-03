import numpy as np
class GradientCheck(object):
	def gradient_check(self,x,theta,epsilon=1e-7):
		thetaplus = theta + epsilon
		thrtaminus = theta - epsilon
		J_plus = forward_propagation(x,thetaplus)
