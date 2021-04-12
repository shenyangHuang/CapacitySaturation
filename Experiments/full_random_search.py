'''
Randomly select 200 architectures and pick the best one
'''
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import datetime
from Model import CNN_Model
from Transformations import CNN_transformation
from Architecture_Embedding import CNNarch_Embedding
from Training import CNN_Training
from Params.param_templates import p_s
from Dataset_Management import datasets
from keras import backend as K 
from Model.Layer_obj import Layer_obj
import tensorflow as tf

import random
import numpy as np
import copy
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def delete_file(filename):
    if(os.path.isfile(filename)):
        os.remove(filename)

def load_file(filename):
    return pickle.load( open( filename, "rb" ) )


def sample_architecture(num_conv, num_fc, num_pooling, conv_list, fc_list, dropout_list):

    #first decide when to insert pooling layer
    if (num_pooling > num_conv):
        num_pooling = num_conv

    pool_idxs = random.sample(range(1,num_conv+1), num_pooling)

    arch = []

    '''
    incrementally append each layer
    '''
    arch.append(Layer_obj([3,3,3,random.choice(conv_list)],'conv'))   #layer 0

    #append conv layers and pooling layers
    for i in range(1,num_conv+1):
        if (i in pool_idxs):
            arch.append(Layer_obj([2],'maxpool'))
            arch.append(Layer_obj([random.choice(dropout_list)],'dropout'))

        if (i != num_conv):
            arch.append(Layer_obj([3,3,3,random.choice(conv_list)],'conv'))


    #append fc layers now
    for i in range(0, num_fc):
        arch.append(Layer_obj([1024,random.choice(fc_list)],'fc'))
        #randomly depend if there is a dropout afterwards
        choices = [True, False]
        decision = random.choice(choices)
        if (decision):
            arch.append(Layer_obj([random.choice(dropout_list)],'dropout'))

    return arch




def learning():
    print ("-----------------------------------------------------")
    print ("starting")
    print(datetime.datetime.now())
    max_score = 0
    searched_archs = {}

    if os.path.isfile('arch_dict.pkl'):
        searched_archs = load_file('arch_dict.pkl')
        values = searched_archs.values()
        for value in values:
            if (value[0] > max_score):
                max_score = value[0]



    '''
    only validate on the first 10 classes
    '''

    num_archs = 200
    order = list(range(100)) 
    classes = datasets.Incremental_partition("cifar100", 0.1, order, normalization="numerical")
    num_classes = 10
    for k in range(1,num_classes):
        datasets.sum_data(classes, k, 0, percentage=1.0)


    '''
    use the same hyperparameters
    '''
    lr = p_s["lr"]
    patience = p_s["patience"]
    batch_size = p_s["batch_size"]
    patience = p_s["patience"]
    epochs = p_s["epochs"]

    #use the same filter size for each conv 
    #append a dropout layer after each pooling layer
    conv_list = list(range(16,512,16))
    fc_list = list(range(64,1024,64)) 
    dropout_list = [0.1,0.25,0.5]

    max_conv = 15
    max_fc = 5
    max_pool = 5


    '''
    decide number of layers
    '''
    for k in range(len(searched_archs), num_archs):
        num_conv = random.randint(1, max_conv)
        num_fc = random.randint(1, max_fc)
        num_pooling = random.randint(0, max_pool)

        arch = sample_architecture(num_conv, num_fc, num_pooling, conv_list, fc_list, dropout_list)
        arch_str = CNNarch_Embedding.get_net_str(arch)
        while (arch_str in searched_archs):
            arch = sample_architecture(num_conv, num_fc, num_pooling, conv_list, fc_list, dropout_list)
            arch_str = CNNarch_Embedding.get_net_str(arch)



        print("----------------------------------------------------")
        print ("architecture "+ str(k))

        cnn_model = CNN_Model.CNN_Model("full retrain")

        (score, cnn_model) = (score, RL_model) = CNN_Training.train_Teacher(cnn_model, classes[0], arch,
        output_dim=num_classes, lr=lr, epochs=epochs, batch_size=batch_size, 
        verbose=False, patience=patience, test=False, class_curve=False, 
        save=False, data_augmentation=False)
        num_params = cnn_model.numParams



        print ("has accuracy " + str(score[1]))
        print ("has param " + str(num_params))
        print ("processing : ")
        print(*arch_str.split('_'), sep = "\n")
        print("----------------------------------------------------")

        cur_score = score[1]

        '''
        adding to dictionary
        '''
        searched_archs[arch_str] = (cur_score, arch, num_params)
        if (cur_score > max_score):
            max_score = cur_score
            #overwrite previous best
            save_object(cnn_model, "best10"+".pkl")

        save_object(searched_archs, "arch_dict"+".pkl")

        K.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

    print ("best validation accuracy achieved is: " + str(max_score))
    print ("ending")
    print(datetime.datetime.now())
    print ("-----------------------------------------------------")



def main():
    learning()



if __name__ == "__main__":
    main()

























































