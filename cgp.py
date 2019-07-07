import numpy as np
import pandas as pd
import cgp_base

class CGPRegressor:
	def __init__(self):
		"no write"

	def fit(self, X, y):
		return self

	def predict(self, X):
		return y

	def get_params(self, deep=True):
		return {}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter, value)
		return self

	def score(self,X,y,sample_weight=None):
		return None

def CGPClassifier:
	def __init__(self):
		"no write"

	def fit(self, X, y):
		return self

	def predict(self, X):
		return y

	def get_params(self, deep=True):
		return {}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self,parameter, value)
		return self

	def score(self,X,y,sample_weight=None):
		return None