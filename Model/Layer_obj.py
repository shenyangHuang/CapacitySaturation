'''
Storing all necessary information for a CNN task network 
only used for transformations
'''

from __future__ import print_function
import numpy as np
import copy




'''
stores both the parameters used by the layer (filter size etc.) as well as what type of layer it is
'''
class Layer_obj:


	'''
	type of layers include:
	'fc'
	'conv'
	'bn'
	'dropout'
	'maxpool'

	
	size is a list indicating related size of a layer such as filter size, stride and so on
	'''

	def __init__(self,size,Ltype):
		self.Ltype = Ltype
		self.size = size
























































