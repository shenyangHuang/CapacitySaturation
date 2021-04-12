'''
Storing all necessary information for a CNN task network 
only used for transformations
'''

from __future__ import print_function
from Model.Layer_obj import Layer_obj
from Architecture_Embedding import CNNarch_Embedding
import numpy as np
import copy




'''
CNN Model now store weights and bias as 
self.weights[0] = weights
self.weights[1] = bias

tasks are a list of newly added identity layers of which its BN layers needs to be initialized by forward pass
'''
class CNN_Model:

	def __init__(self, name):
		self.weights = []
		self.architecture = []
		self.name = name
		#tasks is a list of newly initialized identity layers that needs to account for its BN statistics
		self.tasks = []
		self.numParams = 0

	def export(self):
		return {'weights':copy.deepcopy(self.weights),
				'architecture':copy.deepcopy(self.architecture),
				'tasks':copy.deepcopy(self.tasks),
				'numParams':self.numParams}

	def param_import(self,params):
		self.weights = copy.deepcopy(params['weights'])
		self.architecture = copy.deepcopy(params['architecture'])
		self.tasks = copy.deepcopy(params['tasks'])
		self.numParams = params['numParams']

	def print_arch(self):
		arch_str = CNNarch_Embedding.get_net_str(self.architecture)
		print(*arch_str.split('_'), sep = "\n")


























































