import numpy as np
import re
from Model.Layer_obj import Layer_obj

'''
translate the architecture of a CNN_Model object 
output a string, separated by '-'
'''
def get_net_str(architecture):
    #net_str = ['stop']
    net_str = []
    for layer in architecture:
        if (layer.Ltype == "conv"):
            net_str.append('conv-%d-%d' % (layer.size[0], layer.size[3]))   #filter size, number of channels 
        elif (layer.Ltype == "fc"):
            net_str.append('fc-%d' % layer.size[1]) #number of neurons
        elif (layer.Ltype == "maxpool"):
            net_str.append('maxpool')
        elif (layer.Ltype == "bn"):
            net_str.append('bn')
        elif (layer.Ltype == "dropout"):
            net_str.append('dropout-%f' % layer.size[0])
    return '_'.join(net_str)


'''
translate a network architecture string into a padded list of layers encoding 
return a list of layers encoding with added padding until num_steps
'''
def get_net_seq(net_str, vocabulary, num_steps):
    net_str = re.split('_', net_str)
    net_encoding = vocabulary.get_encoding(net_str)
    net_encoding += [vocabulary.pad_code for _ in range(len(net_encoding), num_steps)]
    return np.array(net_encoding)

'''
Define the range of possible architecture configurations
'''
def define_cnn_vocab():
    '''
    range of configurations
    '''
    filter_num_list = [16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 576, 640]
    units_num_list = [64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]
    kernel_size_list = [1, 3, 5]

    # encoder config
    layer_token_list = ['conv-%d-%d' % (k, f) for f in filter_num_list for k in kernel_size_list]
    layer_token_list += ['fc-%d' % u for u in units_num_list] + ['maxpool'] + ['bn']   #can also add stop

    return Vocabulary(layer_token_list)




'''
Vocabulary for CNN architectures
'''
class Vocabulary:
    def __init__(self, token_list):
        token_list = ['PAD'] + token_list
        self.vocab = {}
        for idx, token in enumerate(token_list):
            self.vocab[token] = idx
            self.vocab[idx] = token
    
    @property
    def size(self):
        return len(self.vocab) // 2

    @property
    def pad_code(self):
        return self.vocab['PAD']

    def get_token(self, encoding_list):
        return [self.vocab[encoding] for encoding in encoding_list]

    def get_encoding(self, token_list):
        return [self.vocab[token] for token in token_list]


    