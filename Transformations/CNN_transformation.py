from __future__ import print_function
from Model import CNN_Model
from Model.Layer_obj import Layer_obj
import numpy as np
import copy
import math 


'''
uses CNN_Model to transform and store a current model
#note: dropout will not be documented in the architecture list 
#now architecture store weight shapes instead of neuron number 
Use Net2WiderNet, Net2DeeperNet, outLayer_transform function provided here
Note Each Convolutional Layer and fully connected layer uses batch normalization
'''

'''
Net2WiderNet of a certain layer
also change the BN layer after that if it exists
the layer number corresponds to a conv or fc layer only
layer number starts from 0 and refers to the architecture layer number
'''
def Net2WiderNet(cnn_model, layer_number, extended_size):

    architecture = cnn_model.architecture
    offset = _countOffset(cnn_model.architecture, layer_number)

    '''
    check if Net2WiderNet can be carried out
    '''
    if(architecture[layer_number].Ltype == 'conv' or architecture[layer_number].Ltype == 'fc'):
        proceed = True
    else:
        raise ValueError('WiderNet can only be applied at conv or fc layers')

    if(extended_size < (architecture[layer_number].size[-1])):
        raise ValueError('WiderNet can only add neurons, extended size must be larger than original')

    if(len(cnn_model.tasks)>0):
        raise ValueError('please Update all newly added BN before proceeding to WiderNet')
    

    hasBN = False
    # if(architecture[layer_number+1].Ltype == 'bn'):
    #     hasBN = True


    '''
    find existing parameters in the model
    '''
    input_weights = cnn_model.weights[layer_number-offset][0]
    input_bias = cnn_model.weights[layer_number-offset][1]
    #check if there is a BN layer in between
    if(hasBN):
        output_weights = cnn_model.weights[layer_number-offset+2][0]
    else:
        output_weights = cnn_model.weights[layer_number-offset+1][0]

    '''
    construct a mapping and add noise to break the symmetry
    '''
    mapping = _WiderNetRemapping(input_weights, output_weights, input_bias, extended_size)
    unit_mapping = mapping['unit_mapping']

    cnn_model.weights[layer_number-offset][0] = _addNoise(mapping['input_remap'], 0.01)
    if(hasBN):
        cnn_model.weights[layer_number-offset+2][0] = _addNoise(mapping['output_remap'], 0.01)
    else:
        cnn_model.weights[layer_number-offset+1][0] = _addNoise(mapping['output_remap'], 0.01)
    cnn_model.weights[layer_number-offset][1] = mapping['bias_remap']


    '''
    Also expand the batch normalization layer immediately following this 
    '''
    if (hasBN):
        cnn_model.weights[layer_number-offset+1] = [Wider_BN(np.asarray(cnn_model.weights[layer_number-offset+1][0]), unit_mapping, extended_size)]
    

    cnn_model.architecture[layer_number].size = list(cnn_model.weights[layer_number-offset][0].shape) 
    if (hasBN):
        cnn_model.architecture[layer_number+1].size = [extended_size]

    return cnn_model


'''
This version only adds identity filter before BN
and doesn't add any new identity filter
'''
def Net2DeeperNet(cnn_model, layer_number, addBN=False):
        
    offset = _countOffset(cnn_model.architecture, layer_number)
    architecture = cnn_model.architecture
    
    hasBN = False

    # if(architecture[layer_number+1].Ltype == 'bn'):
    #     hasBN = True

    if(architecture[layer_number].Ltype == 'conv' or architecture[layer_number].Ltype == 'fc'):
        proceed = True
    else:
        raise ValueError('DeeperNet can only be applied at conv or fc layers')

    '''
    For convolutional layers add identity filter
    '''
    isConv = False
    isFc = False
    if (architecture[layer_number].Ltype =='conv'):
        arch = architecture[layer_number].size
        mid = arch[0] // 2
        identity = np.zeros((arch[0],arch[1],arch[3],arch[3]))
        identity[mid, mid] = np.eye(arch[3])
        num_hidden = arch[-1]
        isConv = True
    else:
        '''
        For FC layers add identity layer
        '''
        num_hidden = (np.asarray(cnn_model.weights[layer_number-offset][0]).shape)[1]
        identity = np.eye(num_hidden)
        isFc = True 

    zero_bias = np.zeros((num_hidden,),dtype='f')
    if(addBN):
        if(not hasBN):
            cnn_model.tasks.append(layer_number-offset+2)
        else:
            cnn_model.tasks.append(layer_number-offset+3)
        cnn_model.tasks.sort()

    new_weights = [] #insert the new weights
    new_weights.append(_addNoise(identity, 0.001))    #break symmetry for Net2DeeperNet
    new_weights.append(zero_bias)
    
    cnn_model.weights.insert(layer_number-offset+1, new_weights)

    #update architecture
    if(isConv):
        cnn_model.architecture.insert(layer_number+1, Layer_obj(list(identity.shape),'conv'))
    if(isFc):
        cnn_model.architecture.insert(layer_number+1, Layer_obj(list(identity.shape),'fc'))
    return cnn_model

'''
transform the output layer to accommodate for more classes
last layer is always FC layer 
'''
def outLayer_transform(cnn_model, new_num, avg=False):
    '''
    find existing parameters
    '''
    past_num = len(cnn_model.weights[-1][1])
    output_weights = cnn_model.weights[-1][0]
    output_bias = cnn_model.weights[-1][1]

    if(past_num > new_num):
        raise ValueError('transforming output layer to smaller sizes are invalid')

    
    if(past_num < new_num):
        '''
        use random initialization
        '''
        if(not avg):
            #using random initialization
            mean = 0.0
            std_dev = 0.35
            weights_random = np.random.normal(mean,std_dev,[len(output_weights),(new_num-past_num)])
            output_weights = np.concatenate((output_weights,weights_random),axis=1)
            cnn_model.weights[-1][0] = output_weights

            bias_random = np.zeros(new_num-past_num)
            output_bias = np.concatenate((output_bias,bias_random))
            cnn_model.weights[-1][1] = output_bias
            '''
            use average from existing weights
            '''
        else:
            average = np.mean(output_weights, axis=1)
            stacked = []
            #note this can be an issue, if it is so, try adding noise, for now, just duplicate average 
            for i in range(new_num-past_num):
                stacked.append(average)
            stacked = np.asarray(stacked,dtype=np.float32)
            stacked = np.reshape(stacked,(len(output_weights),(new_num-past_num)))
            output_weights = np.concatenate((output_weights,stacked),axis=1)
            cnn_model.weights[-1][0] = output_weights

            bias_average = np.mean(output_bias)
            new_bias = []
            for i in range(new_num-past_num):
                new_bias.append(bias_average)
            new_bias = np.asarray(new_bias,dtype=np.float32)
            output_bias = np.concatenate((output_bias,new_bias))
            cnn_model.weights[-1][1] = output_bias

    return cnn_model

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
Able to Remap for both CNN and MLP layers
'''
def _WiderNetRemapping(input_weights, output_weights, input_bias, extended_size):
    #Remapping for CNN
    if(input_weights.ndim>2):
        #map channels 
        original_size = input_weights.shape[3]
        input_mapping = _randomMapping(original_size,extended_size)
        arch = list(copy.deepcopy(input_weights.shape))
        arch[3] = extended_size
        input_remap = np.zeros(arch,dtype="float32")
        #duplicate channels 
        for i in range(0,extended_size):
            input_remap[:,:,:,i] = input_weights[:,:,:,input_mapping[i]]
        occurrence = _occurrence(input_mapping,original_size)
        
        #output weights are Conv layer
        if(output_weights.ndim>2):
            arch = list(copy.deepcopy(output_weights.shape))
            arch[2] = extended_size
            output_remap = np.zeros(arch,dtype="float32")
            for i in range(0, extended_size):
                output_remap[:,:,i,:] = output_weights[:,:,input_mapping[i],:] / occurrence[input_mapping[i]]
        #output layer is a dense layer
        #must pass through a max pool layer and flatten layer 
        else:
            out_size = output_weights.shape[1]
            #first find out the dimension
            dim = int(math.sqrt(output_weights.shape[0]/original_size))
            output_copy = np.reshape(output_weights,(dim,dim,original_size,out_size))
            output_remap = np.zeros((dim,dim,extended_size,out_size),dtype="float32")
            for i in range(0, extended_size):
                output_remap[:,:,i,:] = output_copy[:,:,input_mapping[i],:] / occurrence[input_mapping[i]]
            output_remap = np.reshape(output_remap,(dim*dim*extended_size,out_size))

    #Remapping for MLP
    else:
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

'''
Map BN parameters similar to Net2WiderNet
assume shape = (4,original_size)
extend to (4,extended_size)
'''
def Wider_BN(original_params, unit_mapping, extended_size):
    original_size = original_params.shape[1]
    Shapes = list(original_params.shape)
    Shapes[1] = extended_size
    new_params = np.zeros(Shapes, dtype="float32")
    for i in range(0,extended_size):
        new_params[:,i] = original_params[:,unit_mapping[i]]
    return new_params



'''
We must set the output scale and output bias of the normalization layer to undo the normalization of the layer's statistics
More details see Net2Net paper
statistics stores the statistics from the forward pass
statistics has shape (2,num_hidden)
statistics[0] = moving_mean
statistics[1] = moving_variance
epsilon=0.001 is the default keras epsilon value

order of weights stored in a keras BN layer: [gamma, beta, moving_mean, moving variance]

This function outputs (4,num_hidden)
'''
def set_BN_identity(statistics, epsilon=0.001):
    num_hidden = len(statistics[0])
    new_params = np.zeros((4,num_hidden), dtype="float32")
    new_params[0] = np.sqrt(statistics[1] + epsilon)            #gamma = sqrt(moving_variance + epsilon)
    new_params[1] = statistics[0]       #beta = moving mean
    new_params[2] = statistics[0]       #moving mean
    new_params[3] = statistics[1]       #moving variance
    return new_params



'''
a helper function to count the offset between architecture representation and weights stored inside the cnn_model
This is due to the fact that maxpool layers doesn't have weights
input the layer_number of an architecture and output the number of maxpool layer up until that point
'''
def _countOffset(architecture, layer_number):
    offset = 0
    count = 0
    for layer in architecture:
        if (count > layer_number):
            break
        if(layer.Ltype == 'maxpool'):       #maxpool layer doesn't have weights
            offset = offset + 1
        if(layer.Ltype == 'dropout'):       #dropout layer doesn't have weights
            offset = offset + 1
        count = count + 1
    return offset



def _print_arch(architecture):
    count = 0
    for layer in architecture:
        print ("layer " + str(count) + " is " + layer.Ltype)
        print (layer.size)
        count = count + 1















































