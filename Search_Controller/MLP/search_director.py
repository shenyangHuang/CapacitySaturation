'''
Search Director is in charge of keep the best accuracy and models
As well as evaluating model performances(train them and take validation accuracy)
'''
from __future__ import print_function
from Model import MLP_Model
from Transformations import MLP_transformation
from Architecture_Embedding import CNNarch_Embedding
from Training import MLP_Training
from keras import backend as K 
import tensorflow as tf

import numpy as np
import copy
import pickle

 
class search_director:

    #note that starting_model and top_model will be mlp_model parameters not objects
    def __init__(self, N_workers, lr, epoch_limit, discrete_levels, 
                patience=20):
        self.N_workers = N_workers
        self.lr = lr
        self.epoch_limit = epoch_limit
        self.discrete_levels = discrete_levels
        #shouldn't really use patience in this scenario
        self.patience = patience
        self.starting_model = None
        self.starting_accu = 0.0
        self.top_model = None
        self.top_accu = 0.0
        self.dataset = None
        self.output_dim = 2

    '''
    Must update search parameters for each step
    '''
    def update_step_params(self, starting_model, starting_accu, dataset, output_dim):
        self.starting_model = starting_model
        self.starting_accu = starting_accu
        self.top_model = starting_model
        self.top_accu = starting_accu
        self.dataset = dataset
        self.output_dim = output_dim

    '''
    execute the search process
    instructions for model expansions need to be imported
    each instruction corresponds to a new model configuration
    will report top models and output the best one
    '''
    def search_step(self, instructions, starting_model, starting_accu, dataset, output_dim, RAS=False):
        
        self.update_step_params(starting_model, starting_accu, dataset, output_dim)
        val_accus = [0] * len(instructions)
        #evaluate each sampled model sequentially (for now)
        for i in range(len(instructions)):
            candidate = self.model_evaluation(instructions[i], index=i)
            accuracy = candidate['accuracy']
            val_accus[i] = accuracy
            architecture = candidate['params']['architecture']
            arch_str = CNNarch_Embedding.get_net_str(architecture)
            print ("for architecture " + str(i) + " : ")
            print ("has parametrs : " + str(candidate['params']['numParams']))
            print(*arch_str.split('_'), sep = "\n") 
            print (" achieved validation accuracy of " + str(accuracy))

            if(accuracy > self.top_accu):
                self.top_accu = accuracy
                self.top_model = candidate['params']

        print("Search Completed")
        Expand = self.expansion_decider(val_accus, starting_accu)
        if (RAS):
            Expand = True       #always expand if it is RAS

        if (Expand):
            print("best candidate has the following architecture")
            print("has parameters : " + str(self.top_model['numParams']))
            arch_str = CNNarch_Embedding.get_net_str(self.top_model['architecture'])
            print(*arch_str.split('_'), sep = "\n") 
            print ("best model achieved validation accuracy of " + str(self.top_accu))
        else:
            print ("No expansion for current step")
            self.top_model = starting_model
            self.top_accu = starting_accu
        return ({'accuracy':self.top_accu,'params':self.top_model}, val_accus)
        

    '''
    train a model for a few epochs to evaluate its performance
    '''
    def model_evaluation(self, instruction, index=0): 
        max_accuracy = 0
        output_params = 0
        print ("running the following instruction")
        print (instruction)

        net = MLP_Model.MLP_Model(str(index))
        net.param_import(self.starting_model)
        architecture = net.architecture

        WiderInstructions = instruction['Wider']
        DeeperInstructions = instruction['Deeper']

        if(DeeperInstructions):
        #execute DeeperNet first
            for i in range(len(DeeperInstructions)):
                net = MLP_transformation.Net2DeeperNet(net, DeeperInstructions[i])

        if(WiderInstructions):
            for i in range(len(WiderInstructions)):
                #weights discount max pooling layer and flatten layer 
                curr_width = architecture[WiderInstructions[i]].size[-1]
                #its a dense layer, also increase its capacity to the next discrete level
                if(architecture[WiderInstructions[i]].Ltype == 'fc'):
                    index = self.discrete_levels.index(curr_width)
                    if(index < (len(self.discrete_levels)-1)):
                        next_width = self.discrete_levels[index+1]
                    else:
                        next_width = self.discrete_levels[index]
                    net = MLP_transformation.Net2WiderNet(net, WiderInstructions[i], next_width)

        '''
        Do not run the model if there is no expansion 
        '''
        if(len(DeeperInstructions) == 0 and len(WiderInstructions) == 0):
            accuracy = self.starting_accu

        else:
            (score, net) = MLP_Training.execute(net, self.dataset, self.output_dim,
            lr=self.lr, epochs=self.epoch_limit, batch_size=32,
            verbose=False, patience=self.patience,
            test=False, log_path="none",
            upload=True, class_curve=False, save=False)
            accuracy = score[1]

        output_params = net.export()
        output = {'accuracy':accuracy,'params':output_params}
        '''
        if multiple models are trained at the same time on the same GPU, this might cause a bug
        '''
        # 5. limit gpu usage and doesn't take the entire gpu
        K.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))
        return output


    '''
    a function to decide if expansion is needed, based on 2 principles (and relationship)
    1. number of positive expansion > negative expansion
    2. mean of all expansion > 0
    '''
    def expansion_decider(self, val_list, exist_val):
        val_list = val_list - exist_val
        PosCount = 0
        NegCount = 0
        for val in val_list:
            if val > 0:
                PosCount = PosCount + 1
            else:
                NegCount = NegCount + 1
        print ("number of negative expansions are " + str(NegCount))
        avg = np.mean(val_list)
        print ("mean of all expansions are " + str(avg))
        if (NegCount >= PosCount):
            return False
        if (avg <= 0):
            return False
        return True

    '''
    Nice Utility to have
    '''
    def Persist_Model(self, mlp_model, name):
        self.save_object(mlp_model, name + '.pkl')
        print (name + '.pkl' + 'saved to disk')

    def save_object(self, obj, filename):
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_object(self, filename):
        return pickle.load( open( filename, "rb" ) )
    
    def set_epoch_limit(self, epoch_limit):
        self.epoch_limit = epoch_limit

    def set_lr(self, lr):
        self.lr = lr

    def set_discrete_levels(self, discrete_levels):
        self.discrete_levels = discrete_levels



























































