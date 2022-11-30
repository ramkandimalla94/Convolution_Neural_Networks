import numpy as np

# The Nearest neighbor classifier simply remebers all the training data
class NearestNeighbor:
	def__init__(self):
	pass

	def train(self,X,y):

		""" X is N X D where each row is an example. Y is 1-Dimension of size N """

		self.Xtr = X
		self.ytr = y

	def predict(self,X):

		""" X is N X D where each row is an example we wish to predict label for """

		num_test = X.shape[0]

		# making sure output type matches th input type
		Ypred = np.zeros(num_test, dtype=self.ytr.dtype)


		# looping over test data
		for i in xrange(num_test):

			# find the nearest training image to ith test image
			# using L1 distance (sum of absolute value differences)
			distances = np.sum(np.abs(self.Xtr - X[i,:]),axis = 1)
			min_index = np.argmin(distance) # get the index with smallest distance
			Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

		return Ypred			
