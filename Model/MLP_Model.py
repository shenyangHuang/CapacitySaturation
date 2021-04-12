'''
Storing all necessary information for a MLP task network 
only used for transformations
'''

from __future__ import print_function
from Model.Layer_obj import Layer_obj
import numpy as np
import copy




'''
MLP Model now store weights and bias as 
self.weights[0] = weights
self.weights[1] = bias
'''
class MLP_Model:

	def __init__(self, name):
		self.weights = []
		self.architecture = []
		self.name = name
		self.numParams = 0

	def export(self):
		return {'weights':copy.deepcopy(self.weights),
				'architecture':copy.deepcopy(self.architecture),
				'numParams':self.numParams}

	def param_import(self,params):
		self.weights = copy.deepcopy(params['weights'])
		self.architecture = copy.deepcopy(params['architecture'])
		self.numParams = params['numParams']

	def print_arch(self):
		for layer in self.architecture:
			print ('fc-%d' % layer.size[1])


























































