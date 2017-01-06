project: language classificaiton
created by Jesse Vogel
12/08/2016

Project dependencies:
- python_speech_features: 		https://github.com/jameslyons/python_speech_features
- numpy: 				http://www.numpy.org/
- scipy:				https://docs.scipy.org/doc/scipy-0.14.0/reference/index.html

After installing dependencies, run:
python predict.py "path/to/input/filename" [extract-features generate-model]
	- the optional parameters are not needed to predict the language of an input file. 

Directory Structure:
	progam.py 			<-- source code
	README.txt			
	feature_output / 		<-- extracted feature data
		en.txt
		es.txt
		pl.txt
		(predictions.txt)	<-- input file features (generated after running program once)
	model_output /			<-- generated models 
		model_en.txt			file format:
		model_es.txt			[feature1 #]
		model_pl.txt			[threshhold1 value]
						[feature2 #]
						[threshhold2 value]
						[...]
						[stump1-weight, stump2-weight, ...]
