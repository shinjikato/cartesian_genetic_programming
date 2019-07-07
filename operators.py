import base
import random
from copy import deepcopy
import numpy as np
import pandas as pd

def point_mutate(pop, MUTPB, nodeSet, leafSet):
	function_node = [(name,node_info) for name,node_info in nodeSet.items() if name!="out"]
	max_arg_num = max([node_info["arity"] for node_info in nodeSet.values()])
	for ind in pop:
		for n, node in enumerate(ind):
			if node.arity != 0 and random.random() < MUTPB:
				ind.fitness = None
				if node.func_name == "out":
					node.inputs_arg[0] = random.randrange(ind.function_num)
				else:
					mode = random.choice(["func","arg"])
					if mode == "func":
						name,node_info = random.choice(function_node)
						node.func_name = name
						node.arity = node_info["arity"]
					if mode == "arg":
						i = random.randrange(len(node.arity))
						node.inputs_arg[i] = random.randrange(n)
	return pop

def use_point_mutate(ind, MUTPB, nodeSet, leafSet):
	function_node = [(name,node_info) for name,node_info in nodeSet.items() if name!="out"]
	max_arg_num = max([node_info["arity"] for node_info in nodeSet.values()])
	for ind in pop:
		path = ind.calc_path()
		for n, use in enumerate(path):
			node = ind[n]
			if use and node.arity != 0 and random.random() < MUTPB:
				ind.fitness = None
				if node.func_name == "out":
					node.inputs_arg[0] = random.randrange(ind.function_num)
				else:
					mode = random.choice(["func","arg"])
					if mode == "func":
						name,node_info = random.choice(function_node)
						node.func_name = name
						node.arity = node_info["arity"]
					if mode == "arg":
						i = random.randrange(len(node.arity))
						node.inputs_arg[i] = random.randrange(n)
	return pop

def one_point_crossover(pop, CXPB):
	for i in range(int(len(pop)/2)):
		if random.random() < CXPB:
			indA = pop[i*2]
			indB = pop[i*2+1]
			point = random.randrange(1,len(indA)-1)
			indA[:p], indB[p:] = indB[:p], indA[p:]
			indA.fitness = None
			indB.fitness = None
	return pop

def uniform_crossover(pop, CXPB):
	for i in range(int(len(pop)/2)):
		if random.random() < CXPB:
			indA = pop[i*2]
			indB = pop[i*2+1]
			for n in range(len(indA)):
				if random.random() < 0.5:
					indA[n], indB[n] = indB[n], indA[n]
			indA.fitness = None
			indB.fitness = None
	return indA, indB

def tournament_selection(pop, tour_size):
	pop_size = len(pop)
	pop = [ind for ind in pop if ind.fitness != None]
	pop = [deepcopy(min(random.sample(pop, tour_size),key=lambda ind:ind.fitness)) for n in range(pop_size)]
	return pop

def evaluation(pop, evaluator, evaluator_args):
	for ind in pop:
		fitness = evaluator(pop, *evaluator_args)
		ind.fitness = fitness
	return pop

def makeInitialPopulation(ind_num, nodeSet, leafSet, gene_num, input_num, output_num):
	pop = []
	for _ in range(num_ind):
		ind = base.Individual()
		ind.create(gene_num, input_num, output_num, nodeSet, leafSet)
		pop.append(ind)
	return pop


