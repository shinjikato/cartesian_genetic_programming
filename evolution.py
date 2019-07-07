import random
import numpy as np
import pandas as pd
import base
import operators as op
from copy import deepcopy

def evaluator(ind, X, y, nodeSet, leafSet):
	_y = ind.run(self, X, nodeSet, leafSet)
	return np.sum((y-_y)**2)/len(y)

def GA_evolution(sklean_object):

	ind_num = sklean_object.ind_num
	nodeSet = sklean_object.nodeSet
	leafSet = sklean_object.leafSet
	gene_num = sklean_object.gene_num
	input_num = sklean_object.input_num
	output_num = sklean_object.input_num
	generation_num = sklean_object.generation_num
	tour_size = sklean_object.tour_size

	CXPB = sklean_object.CXPB
	MUTPB = sklean_object.MUTPB
	CXMode = sklean_object.CXMode
	MUTMode = sklean_object.MUTMode

	X = sklean_object.train_X
	y = sklean_object.train_y
	evaluator_args = (X,y,nodeSet,leafSet)

	pop = op.makeInitialPopulation(ind_num, nodeSet, leafSet, gene_num, input_num, output_num)
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
		if MUTMode == "point":
			pop = op.point_mutate(pop, MUTPB, nodeSet, leafSet)
		if MUTMode == "usePoint":
			pop = op.use_point_mutate(pop, MUTPB, nodeSet, leafSet)


	pop = op.evaluation(pop, evaluator, evaluator_args)
	new_elite = min([ind for ind in pop if ind.fitness!=None], key=lambda ind:ind.fitness)
	if elite == None or new_elite.fitness < elite.fitness:
		elite = deepcopy(new_elite)
	return elite

def ES_evolution(sklean_object):
	ind_num = sklean_object.ind_num
	nodeSet = sklean_object.nodeSet
	leafSet = sklean_object.leafSet
	gene_num = sklean_object.gene_num
	input_num = sklean_object.input_num
	output_num = sklean_object.input_num
	generation_num = sklean_object.generation_num

	MUTPB = sklean_object.MUTPB
	MUTMode = sklean_object.MUTMode

	X = sklean_object.train_X
	Y = sklean_object.train_Y
	evaluator_args = (X,y,nodeSet,leafSet)
	
	pop = op.makeInitialPopulation(ind_num, nodeSet, leafSet, gene_num, input_num, output_num)
	elite = None

	for g in range(generation_num):
		pop = op.evaluation(pop, evaluator, evaluator_args)
		new_elite = min([ind for ind in pop if ind.fitness!=None], key=lambda ind:ind.fitness)
		if elite == None or new_elite.fitness < elite.fitness:
			elite = deepcopy(new_elite)
		pop = [deepcopy(elite) for _ in range(ind_num)]
		if MUTMode == "point":
			pop = op.point_mutate(pop, MUTPB, nodeSet, leafSet)
		if MUTMode == "usePoint":
			pop = op.use_point_mutate(pop, MUTPB, nodeSet, leafSet)

	new_elite = min([ind for ind in pop if ind.fitness!=None], key=lambda ind:ind.fitness)
	if elite == None or new_elite.fitness < elite.fitness:
		elite = deepcopy(new_elite)
	return elite