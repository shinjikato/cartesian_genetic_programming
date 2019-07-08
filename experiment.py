from pmlb import fetch_data, regression_dataset_names
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV
import cgp
import random


random.seed(0)
np.random.seed(0)
test_rate = 0.3

for name in regression_dataset_names:
	print(name)
	data = fetch_data(name)
	data = data.sample(frac=1).reset_index(drop=True)
	test_data = data[:int(len(data)*test_rate)]
	train_data = data[int(len(data)*test_rate):]
	
	scaler = StandardScaler()
	scaler.fit(train_data)
	train_data = pd.DataFrame(scaler.transform(train_data), columns=data.columns)
	test_data = pd.DataFrame(scaler.transform(test_data), columns=data.columns)

	train_target = train_data["target"].values
	train_inputs = train_data.drop("target", axis=1).values
	test_target = test_data["target"].values
	test_inputs = test_data.drop("target", axis=1).values


	"""GA search """
	parameters ={
		"ind_num":[500],
		"const_num":[0,1,5,10],
		"gene_num":[10,50,100],
		"generation_num":[200],
		"tour_size":[2,4,8],
		"CXPB":[0.5,0.6,0.7,0.8,0.9,1.0],
		"MUTPB":[0.01,0.03,0.05,0.1,0.2],
		"CXMode":["OnePoint","Uniform"],
		"MUTMode":["Point","UsePoint"],
		"ea_type":["GA"]
	}
	grid_model = GridSearchCV(cgp.CGPRegressor(), parameters, cv=5, n_jobs=-1, error_score=np.nan, scoring=None, verbose=10)
	grid_model.fit(train_inputs, train_target)
	best_params = grid_model.best_params_
	grid_params = grid_model.cv_results_
	ave = 0
	for n in range(5):
		model = CGPRegressor(**best_params)
		mode.fit(train_inputs,train_target)
		ave += model.score(test_inputs,test_target)*-1
	ave = ave/5
	print("  GA best score",ave)

	"""ES search """
	parameters = {
		"ind_num":[4,5,6,7,8,10],
		"const_num":[0,1,5,10],
		"gene_num":[10,50,100],
		"generation_num":[25000],
		"MUTPB":[0.01,0.03,0.05,0.1,0.2],
		"MUTMode":["Point","UsePoint"],
		"ea_type":["ES"],
		"stop_eval":[100000]
	}
	grid_model = GridSearchCV(cgp.CGPRegressor(), parameters, cv=5, n_jobs=-1, error_score=np.nan, scoring=None, verbose=10)
	grid_model.fit(train_inputs, train_target)
	best_params = grid_model.best_params_
	grid_params = grid_model.cv_results_
	ave = 0
	for n in range(5):
		model = CGPRegressor(**best_params)
		mode.fit(train_inputs,train_target)
		ave += model.score(test_inputs,test_target)*-1
	ave = ave/5
	print("  ES best score",ave)
