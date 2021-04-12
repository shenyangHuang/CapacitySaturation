from __future__ import print_function
from Model import MLP_Model
from Model.Layer_obj import Layer_obj
import numpy as np
import copy
import math 


'''
uses MLP_Model to transform and store a current model 
#now architecture store weight shapes  
Use Net2WiderNet, Net2DeeperNet, outLayer_transform function provided here
'''

'''
Net2WiderNet of a fc layer in MLP
'''
def Net2WiderNet(mlp_model, layer_number, extended_size):

    architecture = mlp_model.architecture

    if(extended_size < (architecture[layer_number].size[-1])):
        raise ValueError('WiderNet can only add neurons, extended size must be larger than original')

    '''
    finding existing parameters in the model
    '''
    input_weights = mlp_model.weights[layer_number][0]
    input_bias = mlp_model.weights[layer_number][1]
    output_weights = mlp_model.weights[layer_number+1][0]

    mapping = _WiderNetRemapping(input_weights, output_weights, input_bias, extended_size)


    mlp_model.weights[layer_number][0] = _addNoise(mapping['input_remap'], 0.01)
    mlp_model.weights[layer_number][1] = mapping['bias_remap']
    mlp_model.weights[layer_number+1][0] = _addNoise(mapping['output_remap'], 0.01)
    mlp_model.architecture[layer_number].size = list(mlp_model.weights[layer_number][0].shape)
    return mlp_model

   
'''
Net2DeeperNet of a fc layer in MLP
'''
def Net2DeeperNet(mlp_model, layer_number):
    architecture = mlp_model.architecture
    '''
    For FC layers add identity layer
    '''
    num_hidden = (np.asarray(mlp_model.weights[layer_number][0]).shape)[1]
    identity = np.eye(num_hidden)
    zero_bias = np.zeros((num_hidden,),dtype='f')

    new_weights = [] #insert the new weights
    new_weights.append(_addNoise(identity, 0.001))    #break symmetry for Net2DeeperNet
    new_weights.append(zero_bias)

    mlp_model.weights.insert(layer_number+1, new_weights)
    mlp_model.architecture.insert(layer_number+1, Layer_obj(list(identity.shape),'fc'))
    return mlp_model

'''
transform the output layer to accommodate for more classes
last layer is always FC layer 
'''
def outLayer_transform(mlp_model, new_num):
    '''
    find existing parameters
    '''
    past_num = len(mlp_model.weights[-1][1])
    output_weights = mlp_model.weights[-1][0]
    output_bias = mlp_model.weights[-1][1]

    if(past_num > new_num):
        raise ValueError('transforming output layer to smaller sizes are invalid')

    
    if(past_num < new_num):
        '''
        use random initialization
        '''
        mean = 0.0
        std_dev = 0.35
        weights_random = np.random.normal(mean,std_dev,[len(output_weights),(new_num-past_num)])
        output_weights = np.concatenate((output_weights,weights_random),axis=1)
        mlp_model.weights[-1][0] = output_weights

        bias_random = np.zeros(new_num-past_num)
        output_bias = np.concatenate((output_bias,bias_random))
        mlp_model.weights[-1][1] = output_bias
    return mlp_model

'''
remap n neurons to q neurons
q must > n
'''
def _randomMapping(n,q):
    if q < n:
        raise ValueError("reduce number of neurons in a level is not allowed")
    
    base = np.arange(0, n, 1)
    extend = np.random.randint(0,n,size=(q-n))
    return np.append(base,extend)

def _occurrence(mapping,n):
    base = np.zeros(n)
    for i in range(0,n):
        base[i] = np.count_nonzero(mapping == i)
    return base

'''
add noise to weights to break the symmetry
'''
def _addNoise(input_layer, ratio):
    std = np.std(input_layer) 
    noise = np.random.normal(0, std * ratio, size=input_layer.shape)
    noise = noise.astype('f')
    return input_layer + noise

'''
Remap for MLP layers
'''
def _WiderNetRemapping(input_weights, output_weights, input_bias, extended_size):
    #Remapping for MLP
    prev_size = input_weights.shape[0]
    original_size = input_weights.shape[1]
    out_size = output_weights.shape[1]

    input_mapping = _randomMapping(original_size,extended_size)
    input_remap = np.zeros((prev_size,extended_size), dtype="float32")
    for i in range(0,prev_size):
        for j in range(0,extended_size):
            input_remap[i][j] = input_weights[i][input_mapping[j]]

    occurrence = _occurrence(input_mapping,original_size)
    output_remap = np.zeros((extended_size,out_size),dtype="float32")
    for i in range(0, extended_size):
        for j in range(0, out_size):
            output_remap[i][j] = output_weights[input_mapping[i]][j] / occurrence[input_mapping[i]]

    bias_remap = np.zeros(extended_size,dtype="float32")
    for i in range(0, extended_size):
        bias_remap[i] = input_bias[input_mapping[i]] / occurrence[input_mapping[i]]

    return {'input_remap':input_remap, 'output_remap':output_remap, 'bias_remap':bias_remap, 'unit_mapping':input_mapping}













































