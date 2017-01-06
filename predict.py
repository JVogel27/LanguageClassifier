from learn import generateModel, Stump, Adaboost
from extract import *
import sys

'''=============== PREDICTION PHASE ==============='''

class Model:
	'''
	represents one weighted decision stump
	'''
	def __init__(self, hypotheses, hypotheses_weights):
		self.hypotheses = hypotheses
		self.hypotheses_weights = hypotheses_weights

def predictSample(input_file_name, model_list, n_hypotheses):
	'''
	generate the MFCC features for the input and use it to predict the language
	print out the result
	'''
	calculateMFCC(input_file_name, "./feature_output/prediction.txt")
	loaded_models = []
	for modelName in model_list:
		loaded_models.append(loadModel(modelName, n_hypotheses))
	X = []
	with open("./feature_output/prediction.txt", "r") as file_object:
		for line in file_object:
			if line == '\n':
				continue
			mfcc_features = [float(i) for i in line.split()]
			X.append(mfcc_features)
	results_yes = []
	results_no = []
	for model in loaded_models:
		yes, no = predict(model, X)
		results_yes.append(yes)
		results_no.append(no)

	if max(results_yes) == results_yes[0]:
		print "english"
	elif max(results_yes) == results_yes[1]:
		print "spanish"
	else:
		print "polish"

def loadModel(model_file_name, n_hypotheses):
	'''
	open up the file where the model is stored and convert it to a model object
	'''
	stumps = []
	with open(model_file_name, "r") as file_object:
		list = [float(next(file_object)) for x in xrange(n_hypotheses*2)]
		for i in range(0, len(list), 2):
			stump = Stump(int(list[i]), list[i+1])
			stumps.append(stump)
		line_array = next(file_object).split()
		weights = [float(i) for i in line_array]
	return Model(stumps, weights)

def predict(model, samples):
	'''
	return 1 if it's a positive match, 0 otherwise
	'''
	hypotheses_weights = model.hypotheses_weights
	hypotheses = model.hypotheses
	yes = 0
	no = 0
	for h, w in zip(hypotheses, hypotheses_weights):
		results = []
		for sample in samples:
			results.append(h.predict(sample))
		if sum(results) > 0:
			yes += w
		else:
			no += w
	return yes, no

'''========= MAIN =========='''

if __name__ == "__main__":
	usage = "usage: python lab2.py filename [extract-features generate-model]"
	noFile = "no filename given"
	n_hypotheses = 1 # change this to adjust # of stumps created
	args = sys.argv[1:]
	if len(sys.argv) == 1:
		print noFile
		print usage
		sys.exit()
	if sys.argv[1] != "extract-features" or sys.argv[1] != "generate-model":
		filename = sys.argv[1]
	if "extract-features" in args:
		extractFeatures()
	if "generate-model" in args:
		generateModel("./model_output/model_{0}.txt", n_hypotheses)
	model_list = ["./model_output/model_en.txt", "./model_output/model_es.txt", "./model_output/model_pl.txt"]
	predictSample(filename, model_list, n_hypotheses)