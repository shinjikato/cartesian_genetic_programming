import random
import numpy as np
import pandas as pd
import base
import operators as op
from copy import deepcopy

def evaluator(ind, X, y, nodeSet, leafSet):
	_y = ind.run(X, nodeSet, leafSet)
	return np.sum((y-_y)**2)/len(y)

def GA_evolution(sklearn_object):
	ind_num = sklearn_object.ind_num
	nodeSet = sklearn_object.nodeSet
	leafSet = sklearn_object.leafSet
	gene_num = sklearn_object.gene_num
	output_num = sklearn_object.output_num
	generation_num = sklearn_object.generation_num
	tour_size = sklearn_object.tour_size

	CXPB = sklearn_object.CXPB
	MUTPB = sklearn_object.MUTPB
	CXMode = sklearn_object.CXMode
	MUTMode = sklearn_object.MUTMode

	X = sklearn_object.train_X
	y = sklearn_object.train_y
	evaluator_args = (X,y,nodeSet,leafSet)

	pop = op.makeInitialPopulation(ind_num, nodeSet, leafSet, gene_num, output_num)
	elite = None

	for g in range(generation_num):
		pop = op.evaluation(pop, evaluator, evaluator_args)

		new_elite = min([ind for ind in pop if ind.fitness!=None], key=lambda ind:ind.fitness)
		if elite == None or new_elite.fitness < elite.fitness:
			elite = deepcopy(new_elite)
		pop = op.tournament_selection(pop, tour_size, elite)

		if CXMode == "OnePoint":
			pop = op.one_point_crossover(pop, CXPB)
		if CXMode == "Uniform":
			pop = op.uniform_crossover(pop, CXPB)
		if MUTMode == "Point":
			pop = op.point_mutate(pop, MUTPB, nodeSet, leafSet)
		if MUTMode == "UsePoint":
			pop = op.use_point_mutate(pop, MUTPB, nodeSet, leafSet)


	pop = op.evaluation(pop, evaluator, evaluator_args)
	new_elite = min([ind for ind in pop if ind.fitness!=None], key=lambda ind:ind.fitness)
	if elite == None or new_elite.fitness < elite.fitness:
		elite = deepcopy(new_elite)
	return elite

def ES_evolution(sklearn_object):
	ind_num = sklearn_object.ind_num
	nodeSet = sklearn_object.nodeSet
	leafSet = sklearn_object.leafSet
	gene_num = sklearn_object.gene_num
	output_num = sklearn_object.output_num
	generation_num = sklearn_object.generation_num

	MUTPB = sklearn_object.MUTPB
	MUTMode = sklearn_object.MUTMode

	X = sklearn_object.train_X
	Y = sklearn_object.train_Y
	evaluator_args = (X,y,nodeSet,leafSet)

	pop = op.makeInitialPopulation(ind_num, nodeSet, leafSet, gene_num, output_num)
	elite = None

	for g in range(generation_num):
		pop = op.evaluation(pop, evaluator, evaluator_args)
		new_elite = min([ind for ind in pop if ind.fitness!=None], key=lambda ind:ind.fitness)
		if elite == None or new_elite.fitness < elite.fitness:
			elite = deepcopy(new_elite)
		pop = [deepcopy(elite) for _ in range(ind_num)]
		if MUTMode == "Point":
			pop = op.point_mutate(pop, MUTPB, nodeSet, leafSet)
		if MUTMode == "UsePoint":
			pop = op.use_point_mutate(pop, MUTPB, nodeSet, leafSet)

	new_elite = min([ind for ind in pop if ind.fitness!=None], key=lambda ind:ind.fitness)
	if elite == None or new_elite.fitness < elite.fitness:
		elite = deepcopy(new_elite)
	return elite