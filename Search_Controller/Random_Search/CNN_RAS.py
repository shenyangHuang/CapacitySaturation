from Model import CNN_Model
from Model.Layer_obj import Layer_obj
from Transformations import CNN_transformation
from Training import CNN_Training
import numpy as np
import random
import copy


'''
randomly pick Net2DeeperNet or Net2WiderNet 
Then randomly pick a layer to apply it on 
Skip the Pooling layers
return an instruction for a new layer configuration 
'''

def random_sample(architecture, num_actions, fc_limit):
    architecture = copy.deepcopy(architecture)  
    max_len = len(architecture) - 1 
    Wider = []
    Deeper = []

    num_deeper = num_actions[0]  
    num_wider = num_actions[1]

    '''
    apply deeperNet first because it changes number of layers
    '''
    for i in range(num_deeper):
        (convIdx,fcIdx) = find_layers(architecture)
        '''
        if fc layers haven't reached its limit (as set by fc_limit)
        find from a list of all conv and fc index
        else: only sample from convIdx
        '''
        if (len(fcIdx) >= fc_limit):
            layer_num = random.choice(convIdx)
        else:
            layer_num = random.choice(fcIdx + convIdx)
        Deeper.append(layer_num)
        architecture.insert(layer_num+1, architecture[layer_num])

    for i in range(num_wider):
        (convIdx,fcIdx) = find_layers(architecture)
        layer_num = random.choice(fcIdx + convIdx)
        Wider.append(layer_num)

    return {'Wider':Wider,'Deeper':Deeper}

'''
generate a list of transformation instructions 
max_actions indicates what are the maximum number of transformation actions allowed
max_duplicates indicates after how many duplicates are sampled, the algorithm will stop caring about duplicates (this is possible in early steps) 
'''

def random_search(architecture, sample_size, wider_max_actions, deeper_max_actions, max_duplicate, fc_limit):
    print ("generate " + str(sample_size) + " of random configurations")
    instructions = []
    counter = 0 
    for i in range(sample_size):
        equal = True 
        sample = None
        while(equal):
            equal = False
            '''
            Randomly sample number of Net2WiderNet action and Net2DeeperNet action
            '''
            num_actions = [np.random.randint(wider_max_actions) + 1, np.random.randint(deeper_max_actions) + 1]
            sample = random_sample(architecture, num_actions, fc_limit)
            for instruction in instructions:
                if(instruction == sample):
                    equal = True
                    counter = counter + 1
            if(counter>max_duplicate):
                equal = False
        instructions.append(sample)
    
    return instructions

'''
simply a helper function that finds conv and fc layers in the architecture (these are expandable)
'''
def find_layers(architecture):
    fcIdx = []
    convIdx = []
    for i in range(0,len(architecture)):
        if(architecture[i].Ltype == 'conv'):
            convIdx.append(i)

        if(architecture[i].Ltype == 'fc'):
            fcIdx.append(i)
    return (convIdx, fcIdx)

