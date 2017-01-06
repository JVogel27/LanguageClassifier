from python_speech_features import mfcc 
import scipy.io.wavfile as wav
import numpy as np
import os

'''=============== EXTRACT FEATURE ==============='''

def extractFeatures():
	en_data = {"input_list": ["./input/en19.wav", "./input/en23a.wav", "./input/en23b.wav", "./input/en32.wav"], "output": "./feature_output/en.txt"}
	es_data = {"input_list": ["./input/es1.wav", "./input/es2.wav", "./input/es3.wav"], "output": "./feature_output/es.txt"}
	pl_data = {"input_list": ["./input/pl3.wav", "./input/pl8.wav", "./input/pl26.wav"], "output": "./feature_output/pl.txt"}
	data = [en_data, es_data, pl_data]
	for d in data:
		for i in d["input_list"]:
			calculateMFCC(i, d["output"])

def calculateMFCC(inputfile, outputfile):
	'''
	use the python_speech_features library to comute the MFCC for each sample
	save the samples to a txt file
	''' 
	try:
		os.remove(outputfile)
	except OSError:
		pass
 	output_file_obj = open(outputfile, "a")
	(rate, data) = wav.read(inputfile)   
	mfcc_feat = mfcc(data, rate)
	np.savetxt(output_file_obj, mfcc_feat)
	output_file_obj.write("\n")
 	output_file_obj.close()