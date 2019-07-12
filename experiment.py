from pmlb import fetch_data, regression_dataset_names
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV
import cgp
import random
from joblib import Parallel,delayed

cpu_num = -1
random.seed(0)
np.random.seed(0)
test_rate = 0.3

def process(params,train_inputs,train_target,test_inputs,test_target):
	model = cgp.CGPRegressor(**params)
	model.fit(train_inputs,train_target)
	test_score = model.score(test_inputs,test_target)*-1
	train_score = model.score(train_inputs,train_target)*-1
	return test_score, train_score

for name in regression_dataset_names:
	problem_ret = []
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
		"const_num":[5],
		"gene_num":[100],
		"generation_num":[200],
		"tour_size":[2,4,8],
		"CXPB":[0.5,0.7,0.9],
		#"MUTPB":[0.01,0.05,0.1,0.2],
		"CXMode":["OnePoint","Uniform"],
		"MUTMode":["Point","UsePoint"],
		"ea_type":["GA"]
	}
	grid_model = GridSearchCV(cgp.CGPRegressor(), parameters, cv=5, n_jobs=cpu_num, scoring=None, verbose=10)
	grid_model.fit(train_inputs, train_target)
	best_params = grid_model.best_params_
	grid_params = grid_model.cv_results_
	args = [(best_params,train_inputs,train_target,test_inputs,test_target) for _ in range(5)]

	result = Parallel(n_jobs=cpu_num)([delayed(process)(*params) for params in args])
	ga_test_ave = sum([test_score for test_score ,train_score in result])/5
	ga_train_ave = sum([train_score for test_score ,train_score in result])/5

	#print("  GA best score",ave)

	"""ES search """
	parameters = {
		"ind_num":[4,6,8],
		"const_num":[5],
		"gene_num":[100],
		"generation_num":[25000],
		"MUTPB":[0.01,0.05,0.1],
		"MUTMode":["Point","UsePoint"],
		"ea_type":["ES"],
		"stop_eval":[100000]
	}
	grid_model = GridSearchCV(cgp.CGPRegressor(), parameters, cv=5, n_jobs=cpu_num, error_score=np.nan, scoring=None, verbose=10)
	grid_model.fit(train_inputs, train_target)
	best_params = grid_model.best_params_
	grid_params = grid_model.cv_results_

	args = [(best_params,train_inputs,train_target,test_inputs,test_target) for _ in range(5)]

	result = Parallel(n_jobs=cpu_num)([delayed(process)(*params) for params in args])
	es_test_ave = sum([test_score for test_score ,train_score in result])/5
	es_train_ave = sum([train_score for test_score ,train_score in result])/5

	csv = pd.DataFrame([[ga_train_ave, ga_test_ave, es_train_ave, es_test_ave]],columns=["GA train","GA test","ES train","ES test"])
	out_name = "experiment_out/" + name + ".csv"
	csv.to_csv(out_name)
