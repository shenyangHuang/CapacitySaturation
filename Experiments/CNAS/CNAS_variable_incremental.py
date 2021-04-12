# For reproducibility of the experiments
# However, gpu might still generate stochasticity
seed_value = 1

from tensorflow.python.client import device_lib
print ("recognized GPU")
print(device_lib.list_local_devices())

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
from keras import backend as K
tf.set_random_seed(seed_value)

# 5. limit gpu usage and doesn't take the entire gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

import datetime
import os
import copy 
from Dataset_Management import datasets
from Model import CNN_Model
from Model.Layer_obj import Layer_obj
from Training import CNN_Training
from Search_Controller import search_director
from Search_Controller.RL.RL_director import RL_director
from Measurements.KnaryClassification import knaryAccuracy
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def delete_file(filename):
    if(os.path.isfile(filename)):
        os.remove(filename)

def load_file(filename):
    return pickle.load( open( filename, "rb" ) )

def learning():

    print ('using random seed of value ' + str(seed_value))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sample_size = 30
    wider_max_actions = 6
    deeper_max_actions = 6
    minRange = 2    #minimum number of new classes arrive at each step
    maxRange = 19   #maximum number of new classes arrive at each step 
    total = 100     #total number of classes in cifar100

    print ("Can maximumly take" + str(wider_max_actions-1) + "wider actions for Net2WiderNet")
    print ("Can maximumly take" + str(deeper_max_actions-1) + "deeper actions for both Net2DeeperNet")

    print ("samples " + str(sample_size) + " architectures ")


    #training hyperparameter 
    lr = 0.0001
    epochs = 1000
    patience = 10
    data_augmentation = False
    batch_size = 128
    discrete_levels = (16,32,64,96,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024)
    

    teacher_architecture = []
    teacher_architecture.append(Layer_obj([3,3,3,128],'conv'))
    teacher_architecture.append(Layer_obj([3,3,128,128],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.1],'dropout'))
    teacher_architecture.append(Layer_obj([3,3,64,128],'conv'))
    teacher_architecture.append(Layer_obj([3,3,64,128],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.25],'dropout'))
    teacher_architecture.append(Layer_obj([3,3,128,256],'conv'))
    teacher_architecture.append(Layer_obj([3,3,256,256],'conv'))
    teacher_architecture.append(Layer_obj([2],'maxpool'))
    teacher_architecture.append(Layer_obj([0.5],'dropout'))
    teacher_architecture.append(Layer_obj([1024,256],'fc'))
    teacher_architecture.append(Layer_obj([0.5],'dropout'))



    #architecture search hyperparameter
    N_workers = 1
    epoch_limit = 20
    max_layers = 50         #have up to 50 layers
    num_input = 4			#use 1 float for Vaccu, 1 float for #conv, 1 float for #fc
    fc_limit = 3
    
    entropy_penalty = 0.01
    RL_lr = 0.001
    max_duplicate = 3


    order = list(range(100)) 
    classes = datasets.Incremental_partition("cifar100", 0.1, order, normalization="numerical")
    print ("integer order")
    if(os.path.isfile("ordering.pkl")):
        print ("order loaded")
        order = load_file("ordering.pkl")
    else:
        order = datasets.variable_K_integer(total, minRange, maxRange)
        save_object(order, "ordering.pkl")
    RL_model = CNN_Model.CNN_Model("RL_model")


    #first check if there is existing model
    Mname = "model_"+"step_"+str(0)+".pkl"
    stepNum = 0
    classes_learned = 0
    for i in range(0,100):
        Mname = "model_"+"step_"+str(i)+".pkl"
        if(os.path.isfile(Mname)):
            RL_model = load_file(Mname)     #there should only exist 1 model
            stepNum = i
            for j in range(0,i+1):
                classes_learned = classes_learned + len(order[j])
                for element in order[j]:
                    datasets.sum_data(classes, element, 0)
            break
    if(stepNum == 0):
        Mname = "model_"+"step_"+str(0)+".pkl"
        for k in order[0]:
            datasets.sum_data(classes, k, 0)
        classes_learned = classes_learned + len(order[0])
        print("no existing model found")
        print("training step 0")
        print(datetime.datetime.now())
        print ("at step " + str(0) + " learn these " + str(order[0]) + " classes ")
        (score, RL_model) = CNN_Training.train_Teacher(RL_model, classes[0], teacher_architecture,
                output_dim=classes_learned, lr=lr, epochs=epochs, batch_size=batch_size, 
                verbose=False, patience=patience, test=True, class_curve=False, 
                save=False, data_augmentation=data_augmentation)
        print("[FTA] Teacher network achieved test accuracy " + str(score[1]))
        print("teacher network has " + str(RL_model.numParams) + " parameters")
        save_object(RL_model, Mname)
    else:
        print(Mname + " is found")

    '''
    initialize the search director 
    '''
    search_dir =  search_director.search_director(N_workers, lr, epoch_limit, discrete_levels,
                    patience=patience, data_augmentation=data_augmentation)

    (predictions, pre_score) = CNN_Training.class_evaluation(RL_model, classes[0], output_dim=classes_learned, lr=lr, batch_size=batch_size, verbose=False)
    stepNum = stepNum + 1
    checkpoint = True
    '''
    initialize RL director 
    '''
    RL_dir = RL_director(num_input, wider_max_actions, deeper_max_actions, max_layers, fc_limit, entropy_penalty=entropy_penalty, lr=RL_lr)
    if os.path.isfile('widerActor') and os.path.isfile('deeperActor'):
        RL_dir.load_actors()
    else:
        RL_dir.save_actors()

    numNew = 0


    for i in range(stepNum, len(order)):
        print ("--------------------------------------------------------------")
        print("training step " + str(i))
        print(datetime.datetime.now())
        print ("saturate with new data")
        for k in order[i]:
            datasets.sum_data(classes, k, 0)
        numNew = len(order[i])
        print ("at step " + str(i) + " learn these " + str(order[i]) + " classes ")
        classes_learned = classes_learned + len(order[i])
        log_path = "./logs/"+"step_"+ str(i)+"_RL"
        #VnotTrained = CNN_Training.get_val_accu(RL_model, classes[0], output_dim=classes_learned, lr=lr, batch_size=batch_size, verbose=False)
        (Vexist, RL_model) = CNN_Training.execute(RL_model, classes[0], output_dim=classes_learned,
                                    lr=lr, epochs=epochs, batch_size=batch_size,
                                    verbose=False, patience=patience,test=False, 
                                    log_path=log_path, baseline=False,
                                    upload=True, class_curve=False, save=False, data_augmentation=data_augmentation)
        print(datetime.datetime.now())
        params = RL_model.export()
        print("[PSA]")
        RL_model.print_arch()
        log_path = "./logs/"+"step_"+ str(i)+"_1RL"
        print ("[PVA] before RL search, model achieved validation accuracy of " + str(Vexist[1]))
        #obtain randomly sampled architectures 
        instructions = RL_dir.RL_sample_step(RL_model.architecture, sample_size, Vexist[1], numNew, max_duplicate)
        print("before updating RL controller:")
        print ("[RLO]")
        RL_dir.RL_status(states=None)       #must run after RL_sample_step
        (candidate, val_accus) = search_dir.search_step(instructions, params, Vexist[1], classes[0], classes_learned)
        print(datetime.datetime.now())

        print("update RL controller -----updated predictions")
        RL_dir.RL_update_step(Vexist[1], val_accus, sample_size, instructions=None, states=None)
        print ("[RLS]")
        RL_dir.RL_status(states=None)
        print("RL candidate achieved validation acccuracy of " + str(candidate['accuracy']) + " before further training ")
        RL_model.param_import(candidate['params'])

        (predictions, pre_score) = CNN_Training.class_evaluation(RL_model, classes[0], output_dim=classes_learned,
            lr=lr, batch_size=batch_size, verbose=False)
        print("RL candidate achieved test accuracy " + str(pre_score[1])+ " before further training ")
        preTrainedVal = candidate['accuracy']
        Vtwo = preTrainedVal
        
        RL_copy = copy.deepcopy(RL_model)       #in case we need to revert back after further training
        (score, RL_model) = CNN_Training.execute(RL_model, classes[0], output_dim=classes_learned,
                                    lr=lr, epochs=epochs, batch_size=batch_size,
                                    verbose=False, patience=patience,test=True, 
                                    log_path=log_path, baseline=False,
                                    upload=True, class_curve=False, save=False, data_augmentation=data_augmentation)
        TrainedVal = CNN_Training.get_val_accu(RL_model, classes[0], output_dim=classes_learned, lr=lr, batch_size=batch_size, verbose=False)
        print("[FCA] RL candidate has architecture of:")
        RL_model.print_arch()
        if(TrainedVal < preTrainedVal):
            RL_model = RL_copy
            print(" revert back to previous weights after further training ")
            print("[FTA] RL candidate achieved test accuracy of " + str(pre_score[1]) + " after further training ")
        else:
            Vtwo = TrainedVal
            print("[FTA] RL candidate achieved test accuracy of " + str(score[1]) + " after further training ")

        #search_dir.Persist_Model(predictions, 'predictions' + 'step_'+str(i))           #also persist the predictions from that step
        fname = "model_"+"step_"+str(i)+".pkl"
        save_object(RL_model, fname)
        fdelete = "model_"+"step_"+str(i-1)+".pkl"
        delete_file(fdelete)


        print(datetime.datetime.now())
        print("training step " + str(i) + " completed")
        print("--------------------------------------------------------------")


def main():
    learning()



if __name__ == "__main__":
    main()
