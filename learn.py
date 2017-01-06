from math import log
import numpy as np
import os

'''=============== MODEL GENERATION ==============='''

class Stump:
	'''
	class representation of a decision stump
	'''

	def __init__(self, feature, value):
		self.feature = feature # the column # associated with the feature
		self.value = value     # the threshold value of the stump

	def predict(self, example):
		feature_value = example[self.feature]
		if feature_value < self.value:
			return -1
		else: 
			return 1

	def export(self):
		'''
		return a representation of the stump as a pair. 
		use this method to save the stump to a file .
		'''
		return self.feature, self.value

class Adaboost:
	'''
	class representation of the adaboost model
	'''
	def __init__(self, n_hypotheses, hypotheses = [], hypothesis_weights = []):
		self.n_hypotheses = n_hypotheses
		self.hypotheses = hypotheses
		self.hypothesis_weights = hypothesis_weights

	def fit(self, X, y):
		'''
		take the test data and generate a number of weighted decision stumps
		'''
		N, _ = X.shape
		example_weights = [1.0/N] * N #initialize each weight to 1/N
		self.hypotheses = []
		self.hypothesis_weights = [1.0] * self.n_hypotheses
		for k in range(self.n_hypotheses): #each loop will build one Stump 
			feature, value = bestFeature(X, y, example_weights) # returns a pair (feature, value) that can be used to create a stump
			self.hypotheses.append(Stump(feature, value))
			error = 0
			numWrong = 0.0
			for j in range(N):
				if self.hypotheses[k].predict(X[j].tolist()[0]) != y[j]: #accumulate number of misclassifications
					error = error + example_weights[j]
					numWrong += 1
			for j in range(N):
				if self.hypotheses[k].predict(X[j].tolist()[0]) == y[j]: #lower weight of those correctly classified
					example_weights[j] = example_weights[j] * error/(1 - error)
			example_weights = [float(i)/sum(example_weights) for i in example_weights]
			self.hypothesis_weights[k] = log((1 - error) / error)
			#print "percent of input samples guessed correctly for stump #", k, ": ", 1 - (numWrong/N)

		return self.hypotheses, self.hypothesis_weights

def bestFeature(X, y, example_weights):
	'''
	compare information gains for each feature with it's optimal threshold value
	reutern the best feature value pair
	'''
	example_weights = np.array(np.transpose(np.matrix(example_weights))) # convert into column vector
	best_feature = 0
	best_value = 0
	best_information_gain = 0
	full_data = np.append(np.append(X, y, axis=1), example_weights, axis=1) #add class labels and weights to sample data
	num_total, num_features = X.shape
	entropy_before = entropy(full_data.tolist())
	for feature in range(num_features):
		value, information_gain = bestValue(feature, full_data, entropy_before)
		if information_gain > best_information_gain:
			best_information_gain = information_gain
			best_feature = feature
			best_value = value
	return best_feature, best_value

def bestValue(feature, full_data, entropy_before):
	'''
	find the best threshold value of a given feature and use it to compute the best information gain
	return the value and information gain as a pair
	'''
	max_value = np.amax(full_data, axis=0)[0, feature]
	min_value = np.amin(full_data, axis=0)[0, feature]
	delta = max_value - min_value
	best_value = min_value
	best_information_gain = 0
	num_total, _ = full_data.shape
	for test_value in range(int(min_value)+1, int(max_value), int(delta/10)):
		set1, set2 = divideset(full_data, feature, test_value)
		n_set1 = float(len(set1))
		n_set2 = float(len(set2))
		entropy_after = (n_set1/num_total*entropy(set1)) + (n_set2/num_total*entropy(set2))
		information_gain = entropy_before - entropy_after
		if information_gain > best_information_gain:
			best_information_gain = information_gain
			best_value = test_value
	return best_value, best_information_gain

def divideset(rows, column, value):
	'''
	divide a set on a specific column
	'''
	set1 = []
	set2 = []
	for row in rows:
		row = row.tolist()[0]
		if row[column] >= value:
			set1.append(row)
		else:
			set2.append(row)
	return set1, set2

def sumWeights(rows, match):
	'''
	sum the weights of each sample whose class label matches match
	'''
	n_rows = len(rows)
	n_cols = len(rows[0])
	sum = 0
	if n_rows == 0:
		return sum
	for row in rows:
		if int(row[-2]) == match: 
			sum += row[-1]
	return sum

def entropy(rows):
	'''
	calulate entropy using the weighted samples
	'''
	if len(rows) == 0:
		return 0
	weight_pos = sumWeights(rows, 1)
	weight_neg = sumWeights(rows, -1)
	weight_all = weight_neg + weight_pos
	p_neg = weight_neg/weight_all
	p_pos = weight_pos/weight_all
	log_p_neg = 0
	log_p_pos = 0
	if p_neg != 0:
		log_p_neg = log(p_neg, 2)
	if p_pos != 0:
		log_p_pos = log(p_pos, 2)
	return -p_pos*log_p_pos - p_neg*log_p_neg

def preprocessInput(classname):
	'''
	conver the saved text data to a numpy matrix
	'''
	X = []
	y = []
	files = ["en", "es", "pl"]
	for fname in files:
		with open("./feature_output/{0}.txt".format(fname), "r") as file_object:
			for line in file_object:
				mfcc_features = [float(i) for i in line.split()]
				X.append(mfcc_features)
				if fname == classname:
					y.append(1)
				else:
					y.append(-1)
	y = np.matrix(y)
	return (np.matrix(X), np.transpose(y))

def generateModel(outputfile, n_hypotheses):
	'''
	'learn' each model to recognize a specific language
	save the models to a textfile to be used later
	'''
	for lang in ["en", "es", "pl"]:
		#print "Generate Model", lang
		outputfile_copy = outputfile
		(X, y) = preprocessInput(lang)
		adaBoost = Adaboost(n_hypotheses)
		hypotheses, hypotheses_weights =  adaBoost.fit(X, y)
		outputfile_lang = outputfile_copy.format(lang)
		try:
			os.remove(outputfile_lang)
		except OSError:
			pass
		output_file_obj = open(outputfile_lang, "a")
		for h in hypotheses:
			feature, value = h.export()
	  		output_file_obj.write("%d\n%d\n " % (feature, value))
	  	for hw in hypotheses_weights:
	  		output_file_obj.write("%s " % hw)
	  	output_file_obj.close()