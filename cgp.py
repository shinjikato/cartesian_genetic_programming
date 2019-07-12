import numpy as np
import pandas as pd
import cgp_base
import evolution as ea
import random

class CGPRegressor:
	def __init__(self,ind_num=4, nodeSet=None,
		const_num=0, gene_num=100, generation_num=100, tour_size=7,
		CXPB=0.7, CXMode="OnePoint", MUTPB=0.05, MUTMode="UsePoint",
		ea_type="ES",stop_eval=100000):
		self.ind_num = ind_num
		if nodeSet == None:
			self.nodeSet = cgp_base.CreateNodeSet({"add","sub","mul","div"})
		else:
			self.nodeSet = nodeSet
		self.const_num = const_num
		self.gene_num = gene_num
		self.generation_num = generation_num
		self.tour_size = tour_size

		self.CXPB = CXPB
		if ea_type == "GA":
			self.MUTPB = 1.0 - CXPB
		else:
			self.MUTPB = MUTPB
		self.CXMode = CXMode
		self.MUTMode = MUTMode
		self.ea_type = ea_type
		self.output_num = 1
		self.stop_eval = stop_eval

	def fit(self, X, y):
		self.train_X = X
		self.train_y = y
		variable_num= X.shape[1]
		output_num = 1
		self.leafSet = cgp_base.CreateLeafSet(variable_num, self.const_num)
		if self.ea_type == "ES":
			self.best_ind = ea.ES_evolution(self)
		if self.ea_type == "GA":
			self.best_ind = ea.GA_evolution(self)
		return self

	def predict(self, X):
		_y = self.best_ind.run(X, self.nodeSet, self.leafSet)[0]
		return _y

	def get_params(self, deep=True):
		return {
		"nodeSet":self.nodeSet,
		"ind_num":self.ind_num,
		"const_num":self.const_num,
		"gene_num":self.gene_num,
		"generation_num":self.generation_num,
		"tour_size":self.tour_size,
		"CXPB":self.CXPB,
		"MUTPB":self.MUTPB,
		"CXMode":self.CXMode,
		"MUTMode":self.MUTMode,
		"ea_type":self.ea_type
		}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter, value)
		return self

	def score(self,X,y,sample_weight=None):
		_y = self.predict(X)
		s =  np.sum((y-_y)**2)/len(y)
		return -1*s

if __name__ == "__main__":# test code
	seed = 6
	inputs_num = 3
	const_num = 1
	instances_num = 5

	random.seed(seed)
	np.random.seed(seed)
	X = np.random.random((instances_num, inputs_num))
	y = np.random.random(instances_num)

	model = CGPRegressor(ind_num=4, nodeSet=None,
		const_num=const_num, gene_num=100, generation_num=10000, tour_size=2,
		CXPB=0.7, CXMode="OnePoint", MUTPB=0.05, MUTMode="UsePoint",
		ea_type="ES")
	model.fit(X,y)
	s = model.score(X,y)
	print(s)
