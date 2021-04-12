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
from Dataset_Management import datasets
from Model import CNN_Model
from Model.Layer_obj import Layer_obj
from Params import arch_templates
from Params.param_templates import p_s
from Training import CNN_Training
from Search_Controller import search_director
from Search_Controller.Random_Search import CNN_RAS
from Measurements.KnaryClassification import knaryAccuracy
import pickle
import copy 


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

    sample_size = 50
    wider_max_actions = 11
    deeper_max_actions = 6

    print ("Can maximumly take" + str(wider_max_actions) + "wider actions for Net2WiderNet")
    print ("Can maximumly take" + str(deeper_max_actions) + "deeper actions for both Net2DeeperNet")
    print ("samples " + str(sample_size) + " architectures ")

    #training parameter
    lr = p_s["lr"]
    epochs = p_s["epochs"]
    patience = p_s["patience"]
    data_augmentation = False
    batch_size = p_s["batch_size"]
    discrete_levels = p_s["discrete_levels"]

    #arch search hyperparameter
    N_workers = p_s["N_workers"]
    epoch_limit = p_s["epoch_limit"]
    fc_limit = p_s["fc_limit"]

    teacher_architecture = arch_templates.static_arch()

    
    order = list(range(100)) 
    classes = datasets.Incremental_partition("cifar100", 0.1, order, normalization="numerical")
    RAS = CNN_Model.CNN_Model("RAS")


    #first check if there is existing model
    Mname = "model"+str(10)+".pkl"
    startNum = 10
    for i in range(10,100):
        Mname = "model"+str(i)+".pkl"
        if(os.path.isfile(Mname)):
            RAS = load_file(Mname)     #there should only exist 1 model
            startNum = i
            for k in range(1,i):
                datasets.sum_data(classes, k, 0, percentage=1.0)
            print ("prepared dataset for up to " + str(i) + " classes")
            break
    if(startNum == 10):
        for k in range(1,10):
            datasets.sum_data(classes, k, 0, percentage=1.0)
        print("no existing model found")
        print("training step 0")
        print(datetime.datetime.now())
        (score, RAS) = CNN_Training.train_Teacher(RAS, classes[0], teacher_architecture,
                output_dim=startNum, lr=lr, epochs=epochs, batch_size=batch_size, 
                verbose=False, patience=patience, test=True, class_curve=False, 
                save=False, data_augmentation=data_augmentation)
        print("[FTA] Teacher network achieved test accuracy " + str(score[1]))
        print("Teacher Network has " + str(RAS.numParams) + " parameters")
        save_object(RAS, "model"+str(10)+".pkl")
    else:
        print(Mname + " is found")




    '''
    initialize the search director 
    '''
    search_dir =  search_director.search_director(N_workers, lr, epoch_limit, discrete_levels,
                    patience=patience, data_augmentation=data_augmentation)

    for i in range(int(startNum/10),10):
        print ("--------------------------------------------------------------")
        print("training step " + str((i+1)*10))
        print(datetime.datetime.now())
        print ("saturate with new data")
        for k in range(i*10, (i+1)*10):
            datasets.sum_data(classes, k, 0, percentage=1.0)        #add 10 new classes at a time
        log_path = "./logs/"+str((i+1)*10)+"_1RAS"
        (Vexist, RAS) = CNN_Training.execute(RAS, classes[0], output_dim=((i+1)*10),
                                    lr=lr, epochs=epochs, batch_size=batch_size,
                                    verbose=False, patience=patience,test=False, 
                                    log_path=log_path, baseline=False,
                                    upload=True, class_curve=False, save=False, data_augmentation=data_augmentation)
        print(datetime.datetime.now())
        params = RAS.export()
        print("[PSA]")
        RAS.print_arch()
        log_path = "./logs/"+str((i+1)*10)+"_1RAS"
        print ("[PVA] before RAS search, model achieved validation accuracy of " + str(Vexist[1]))
        #obtain randomly sampled architectures 
        instructions = CNN_RAS.random_search(RAS.architecture, sample_size, wider_max_actions, deeper_max_actions, sample_size, fc_limit)
        (candidate, val_accus) = search_dir.search_step(instructions, params, Vexist[1], classes[0], (i+1)*10, RAS=True)
        print(datetime.datetime.now())

        # if (candidate['accuracy'] > Vexist[1]):        #only use the candidate if it is better than existing architecture in val accu
        #     RAS.param_import(candidate['params'])
        # else:
        #     print("RAS candidate is worse than existing architecture, keep existing architecture")
        RAS.param_import(candidate['params'])
        print("[PSA]")
        RAS.print_arch()
        print("RAS candidate achieved validation acccuracy of " + str(candidate['accuracy']) + " before further training ")
        preTrainedVal = candidate['accuracy']
        (predictions, pre_score) = CNN_Training.class_evaluation(RAS, classes[0], output_dim=((i+1)*10),
                                                                lr=lr, batch_size=batch_size, verbose=False)
        print("RAS candidate achieved test accuracy of " + str(pre_score[1]) + " before further training ")
        #further training for RAS
        RAS_copy = copy.deepcopy(RAS)       #in case we need to revert back after further training
        (score, RAS) = CNN_Training.execute(RAS, classes[0], output_dim=((i+1)*10),
                                    lr=lr, epochs=epochs, batch_size=batch_size,
                                    verbose=False, patience=patience,test=True, 
                                    log_path=log_path, baseline=False,
                                    upload=True, class_curve=False, save=False, data_augmentation=data_augmentation)
        TrainedVal = CNN_Training.get_val_accu(RAS, classes[0], output_dim=((i+1)*10), lr=lr, batch_size=batch_size, verbose=False)

        print("[FCA] RAS candidate has architecture of:")
        RAS.print_arch()
        if(TrainedVal < preTrainedVal):
            RAS = RAS_copy
            print(" revert back to previous weights after further training ")
            print("[ICA]")
            print("[FTA] RAS candidate achieved test accuracy of " + str(pre_score[1]) + " after further training ")

        else:
            print("[ICA]")
            print("[FTA] RAS candidate achieved test accuracy of " + str(score[1]) + " after further training ")

        #search_dir.Persist_Model(predictions, 'predictions' + str(((i+1)*10)))           #also persist the predictions from that step
        print(datetime.datetime.now())
        fname = "model"+str(((i+1)*10))+".pkl"
        save_object(RAS, fname)
        fdelete = "model"+str(i*10)+".pkl"
        delete_file(fdelete)

        print(datetime.datetime.now())
        print("training step " + str(((i+1)*10)) + " completed")
        print("--------------------------------------------------------------")


def main():
    learning()



if __name__ == "__main__":
    main()
