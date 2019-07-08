import numpy as np
from copy import deepcopy
import random

def add(stack, args):
	return stack[args[0]]+stack[args[1]]

def sub(stack, args):
	return stack[args[0]]-stack[args[1]]

def mul(stack, args):
	return stack[args[0]]-stack[args[1]]

def div(stack, args):
	return stack[args[0]]-stack[args[1]]

def sin(stack, args):
	return np.sin(stack[args[0]])

def cos(stack, args):
	return np.cos(stack[args[0]])

def exp(stack, args):
	return np.exp(stack[args[0]])

def log(stack, args):
	return np.log(stack[args[0]])

def pass_func(stack, args):
	return stack[args[0]]

def CreateNodeSet(useNodes):
	nodeSet = {}
	if "add" in useNodes:
		nodeSet["add"] = {"function":add, "arity":2, "const":None}
	if "sub" in useNodes:
		nodeSet["sub"] = {"function":sub, "arity":2, "const":None}
	if "mul" in useNodes:
		nodeSet["mul"] = {"function":mul, "arity":2, "const":None}
	if "div" in useNodes:
		nodeSet["div"] = {"function":div, "arity":2, "const":None}
	if "sin" in useNodes:
		nodeSet["sin"] = {"function":sin, "arity":1, "const":None}
	if "cos" in useNodes:
		nodeSet["cos"] = {"function":cos, "arity":1, "const":None}
	if "exp" in useNodes:
		nodeSet["exp"] = {"function":exp, "arity":1, "const":None}
	if "log" in useNodes:
		nodeSet["log"] = {"function":log, "arity":1, "const":None}
	nodeSet["out"] =  {"function":pass_func, "arity":1, "const":None}
	return nodeSet

def CreateLeafSet(variable_num, erc_num):
	leafSet = {}
	for i in range(variable_num):
		leafSet["x{}".format(i)] = {"function":lambda X, n:X[:,n], "arity":0, "const":i}
	for i in range(erc_num):
		leafSet["c{}".format(i)] = {"function":lambda const, N: const*np.ones(N), "arity":0, "const":None}
	return leafSet

class Gene:
	def __init__(self, func_name, inputs_arg, arity, const):
		self.func_name = func_name
		self.inputs_arg = inputs_arg
		self.arity = arity
		self.const = const
	def __str__(self, phenotype=False):
		if self.const == None:
			if phenotype:
				return "name : {}, args : {}".format(self.func_name, self.inputs_arg[:self.arity])
			else:
				return "name : {}, args : {}".format(self.func_name, self.inputs_arg)
		else:
			if "x" in self.func_name:
				return "name : {:<3}, const : {}".format(self.func_name, self.const)
			else:
				return "name : {:<3}, const : {}".format(self.func_name, self.const)

	def __deepcopy__(self, memo):
		new = self.__class__(deepcopy(self.func_name), deepcopy(self.inputs_arg), self.arity, self.const)
		return new


class Individual(list):
	def __init__(self, content=[]):
		list.__init__(self, content)
		self.fitness = None

	def create(self, gene_num, output_num, nodeSet, leafSet, ERC_func=lambda:random.random()*2-1):
		self.output_num = output_num
		variable_leaf = [(name,leaf_info) for name,leaf_info in leafSet.items() if leaf_info["const"]!=None]
		variable_leaf.sort(key=lambda obj:obj[1]["const"])
		const_leaf = [(name,leaf_info) for name,leaf_info in leafSet.items() if leaf_info["const"]==None]
		const_leaf.sort(key=lambda obj:obj[0])

		for n, item in enumerate(variable_leaf):
			name, leaf_info = item
			self.append(Gene(name, [], 0, leaf_info["const"]))
		for n, item  in enumerate(const_leaf):
			name, leaf_info = item
			self.append(Gene(name, [], 0, ERC_func()))

		function_node = [(name,node_info) for name,node_info in nodeSet.items() if name!="out"]
		max_arg_num = max([node_info["arity"] for node_info in nodeSet.values()])
		for n in range(gene_num):
			name,node_info = random.choice(function_node)
			inputs_arg = [random.randrange(len(self)) for x in range(max_arg_num)]
			self.append(Gene(name, inputs_arg, node_info["arity"], None))

		function_num = len(self)
		self.function_num = function_num
		for n in range(output_num):
			inputs_arg = [random.randrange(function_num) for x in range(max_arg_num)]
			self.append(Gene("out", inputs_arg, 1, None))

	def calc_path(self):
		path = [False for n in range(len(self))]
		for n in range(len(self))[::-1]:
			gene = self[n]
			if gene.func_name == "out":
				path[gene.inputs_arg[0]] = True
				path[n] = True
			else:
				if path[n]:
					for i in range(gene.arity):
						path[gene.inputs_arg[i]] = True
		return path

	def run(self, X, nodeSet, leafSet):
		path = self.calc_path()
		calc_rets = [None for _ in range(len(self))]
		for n, use in enumerate(path):
			if use:
				gene = self[n]
				name = gene.func_name
				if gene.arity == 0:
					if "x" in name:
						ret = leafSet[name]["function"](X, gene.const)
					else:
						ret = leafSet[name]["function"](gene.const, len(X))
				else:
					ret = nodeSet[name]["function"](calc_rets, gene.inputs_arg)
				calc_rets[n] = ret
		for n in range(1,self.output_num+1):
			ret = calc_rets[-1*n]
			mask = np.isfinite(ret)
			m = np.mean(ret[mask])
			ret[mask==False] = m
			calc_rets[-1*n] = ret

		return calc_rets[-1*self.output_num:]


	def __deepcopy__(self, memo):
		new = self.__class__(self)
		new[:] = [deepcopy(node) for node in self]
		new.fitness = self.fitness
		new.function_num = self.function_num
		new.output_num = self.output_num
		return new

	def __str__(self, phenotype=False):
		s = ""
		for n,node in enumerate(self):
			s += "{:0>3} {} \n".format(n, node.__str__(phenotype))
		return s

if __name__ == "__main__":# test code
	seed = 6
	inputs_num = 3
	const_num = 1
	instances_num = 5

	random.seed(seed)
	np.random.seed(seed)
	nodeSet = CreateNodeSet({"add","sub","mul","div","sin","cos"})
	leafSet = CreateLeafSet(inputs_num, const_num)
	ind = Individual()
	ind.create(10, len(leafSet), 1, nodeSet, leafSet)
	print(ind.__str__(phenotype=True))
	ind2 = deepcopy(ind)
	print(ind2.__str__(phenotype=True))

	X = np.random.random((instances_num, inputs_num))
	print(X)
	ret = ind.run(X,nodeSet,leafSet)
	print(ret)




