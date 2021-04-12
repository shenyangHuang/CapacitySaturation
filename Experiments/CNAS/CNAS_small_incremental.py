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



from time import process_time 
import os
import copy 
from Dataset_Management import datasets
from Model import CNN_Model
from Params import arch_templates
from Params.param_templates import p_s
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
    FTA_scores = []
    FTA_params = []
    FTA_time = []

    print ('using random seed of value ' + str(seed_value))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sample_size = 20
    wider_max_actions = 4
    deeper_max_actions = 4

    print ("Can maximumly take" + str(wider_max_actions-1) + " wider actions for Net2WiderNet")
    print ("Can maximumly take" + str(deeper_max_actions-1) + " deeper actions for both Net2DeeperNet")
    print ("samples " + str(sample_size) + " architectures ")


    #training hyperparameter 
    lr = p_s["lr"]
    epochs = p_s["epochs"]
    patience = p_s["patience"]
    data_augmentation = False
    batch_size = p_s["batch_size"]
    discrete_levels = p_s["discrete_levels"]
    teacher_architecture = arch_templates.SA_small()

    #architecture search hyperparameter
    N_workers = p_s["N_workers"]
    epoch_limit = p_s["epoch_limit"]
    max_layers = p_s["max_layers"]         #have up to 50 layers
    num_input = p_s["num_input"]			#use 1 float for Vaccu, 1 float for #conv, 1 float for #fc
    fc_limit = p_s["fc_limit"]
    
    entropy_penalty = p_s["entropy_penalty"]
    RL_lr = p_s["RL_lr"]
    max_duplicate = p_s["max_duplicate"]


    order = list(range(100)) 
    classes = datasets.Incremental_partition("cifar100", 0.1, order, normalization="numerical")
    RL_model = CNN_Model.CNN_Model("RL_model")



    #first check if there is existing model
    initNum = 2         #10 base knowledge class
    startNum = initNum      #can resume from any number of classes 
    increment = 2       #add in 2 new classes at a time
    Mname = "model"+str(initNum)+".pkl"
    last_time = 0


    for i in range(initNum,100):
        Mname = "model"+str(i)+".pkl"
        if(os.path.isfile(Mname)):
            RL_model = load_file(Mname)     #there should only exist 1 model
            startNum = i
            for k in range(1,i):
                datasets.sum_data(classes, k, 0, percentage=1.0)
            #print ("prepared dataset for up to " + str(i) + " classes")   
            break
    if(startNum == initNum):
        for k in range(1,startNum):
            datasets.sum_data(classes, k, 0, percentage=1.0)
        print("no existing model found")
        print("training step 0")
        print(process_time())
        (score, RL_model) = CNN_Training.train_Teacher(RL_model, classes[0], teacher_architecture,
                output_dim=startNum, lr=lr, epochs=epochs, batch_size=batch_size, 
                verbose=False, patience=patience, test=True, class_curve=False, 
                save=False, data_augmentation=data_augmentation)
        print("[FTA] Teacher network achieved test accuracy " + str(score[1]))
        save_object(RL_model, "model"+str(startNum)+".pkl")
        FTA_scores.append(score[1])
        FTA_params.append(RL_model.numParams)
        FTA_time.append(process_time())
    else:
        print(Mname + " is found")


    '''
    initialize the search director 
    '''
    search_dir =  search_director.search_director(p_s["N_workers"], p_s["lr"], p_s["epoch_limit"], p_s["discrete_levels"],
                    p_s["batch_size"], patience=p_s["patience"], data_augmentation=p_s["data_augmentation"])

    '''
    initialize RL director 
    '''
    RL_dir = RL_director(num_input, wider_max_actions, deeper_max_actions, max_layers, fc_limit, entropy_penalty=entropy_penalty, lr=RL_lr)
    if os.path.isfile('widerActor') and os.path.isfile('deeperActor'):
        RL_dir.load_actors()
    else:
        RL_dir.save_actors()

    for i in range(int(startNum/increment),int(100/increment)):
        print ("--------------------------------------------------------------")
        print("training step " + str((i+1)*increment))
        print(process_time())
        print ("saturate with new data")
        for k in range(i*increment, (i+1)*increment):
            datasets.sum_data(classes, k, 0, percentage=1.0)        #add 2 new classes at a time
        log_path = "./logs/"+str((i+1)*increment)+"_1RL"
        #VnotTrained = CNN_Training.get_val_accu(RL_model, classes[0], output_dim=((i+1)*increment), lr=lr, batch_size=batch_size, verbose=False)
        (Vexist, RL_model) = CNN_Training.execute(RL_model, classes[0], output_dim=((i+1)*increment),
                                    lr=lr, epochs=epochs, batch_size=batch_size,
                                    verbose=False, patience=patience,test=False, 
                                    log_path=log_path, baseline=False,
                                    upload=True, class_curve=False, save=False, data_augmentation=data_augmentation)
        params = RL_model.export()
        print("[PSA]")
        RL_model.print_arch()
        log_path = "./logs/"+str((i+1)*increment)+"_1RL"
        print ("[PVA] before RL search, model achieved validation accuracy of " + str(Vexist[1]))
        instructions = RL_dir.RL_sample_step(RL_model.architecture, sample_size, Vexist[1], increment, max_duplicate)
        print("before updating RL controller:")
        print ("[RLO]")
        RL_dir.RL_status(states=None)       #must run after RL_sample_step
        (candidate, val_accus) = search_dir.search_step(instructions, params, Vexist[1], classes[0], (i+1)*increment, RAS=False, valMemory=None, memoryBuffer=None)

        print("update RL controller -----updated predictions")
        RL_dir.RL_update_step(Vexist[1], val_accus, sample_size, instructions=None, states=None)
        print ("[RLS]")
        RL_dir.RL_status(states=None)
        print("RL candidate achieved validation acccuracy of " + str(candidate['accuracy']) + " before further training ")
        RL_model.param_import(candidate['params'])

        (predictions, pre_score) = CNN_Training.class_evaluation(RL_model, classes[0], output_dim=((i+1)*increment),
            lr=lr, batch_size=batch_size, verbose=False)
        print("RL candidate achieved test accuracy " + str(pre_score[1])+ " before further training ")
        preTrainedVal = candidate['accuracy']
        Vtwo = preTrainedVal
        
        #further training for RL_model
        RL_copy = copy.deepcopy(RL_model)       #in case we need to revert back after further training
        (score, RL_model) = CNN_Training.execute(RL_model, classes[0], output_dim=((i+1)*increment),
                                    lr=lr, epochs=epochs, batch_size=batch_size,
                                    verbose=False, patience=patience,test=True, 
                                    log_path=log_path, baseline=False,
                                    upload=True, class_curve=False, save=False, data_augmentation=data_augmentation)

        TrainedVal = CNN_Training.get_val_accu(RL_model, classes[0], output_dim=((i+1)*increment), lr=lr, batch_size=batch_size, verbose=False)
        print("[FCA] RL candidate has architecture of:")
        RL_model.print_arch()
        FTA_params.append(RL_model.numParams)
        if(TrainedVal < preTrainedVal):
            RL_model = RL_copy
            print(" revert back to previous weights after further training ")
            print("[FTA] RL candidate achieved test accuracy of " + str(pre_score[1]) + " after further training ")
            FTA_scores.append(pre_score[1])
        else:
            Vtwo = TrainedVal
            print("[FTA] RL candidate achieved test accuracy of " + str(score[1]) + " after further training ")
            FTA_scores.append(score[1])


        #search_dir.Persist_Model(predictions, 'predictions' + str((i+1)*increment))           #also persist the predictions from that step
        fname = "model"+str((i+1)*increment)+".pkl"
        save_object(RL_model, fname)
        fdelete = "model"+str((i)*increment)+".pkl"
        delete_file(fdelete)

        FTA_time.append(process_time())
        print("training step " + str((i+1)*increment) + " completed")
        print("--------------------------------------------------------------")
        print ("all test scores are:")
        print (FTA_scores)
        print("--------------------------------------------------------------")
        print ("all architecture parameters are:")
        print (FTA_params)
        print("--------------------------------------------------------------")
        print ("all time differences are:")
        print (FTA_time)




def main():
    learning()



if __name__ == "__main__":
    main()
