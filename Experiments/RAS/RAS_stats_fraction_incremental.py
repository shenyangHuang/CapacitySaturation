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

    sample_size = 5
    wider_max_actions = 4
    deeper_max_actions = 4
    initial_classes = [0,1,2,3,4,5,6,7,8,9]     #start with 10 classes
    total = 100
    minRange = 0
    maxRange = 0
    fractionList = [0.5]


    print ("Can maximumly take" + str(wider_max_actions-1) + " wider actions for Net2WiderNet")
    print ("Can maximumly take" + str(deeper_max_actions-1) + " deeper actions for both Net2DeeperNet")
    print ("samples " + str(sample_size) + " architectures ")

    #training parameter
    lr = 0.0001
    epochs = 1000
    patience = 10
    data_augmentation = False
    batch_size = 128
    discrete_levels = (16,32,64,96,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024)

    #same architecture as 10 class incremental learning
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
    fc_limit = 3
    


    order = list(range(100)) 
    classes = datasets.Incremental_partition("cifar100", 0.1, order, normalization="numerical")
    print ("fraction order")
    if(os.path.isfile("ordering.pkl")):
        order = load_file("ordering.pkl")
    else:
        order = datasets.variable_K_fraction(total, fractionList, minRange, maxRange, initial_classes=initial_classes)
        print ("ordering of class arrival is:")
        for element in order:
            print (element)
        save_object(order, "ordering.pkl")
    RAS = CNN_Model.CNN_Model("RAS")


    #first check if there is existing model
    Mname = "model_"+"step_"+str(0)+".pkl"
    stepNum = 0
    classes_learned = 0
    frac_class = 0
    for i in range(0,100):
        Mname = "model_"+"step_"+str(i)+".pkl"
        if(os.path.isfile(Mname)):
            RAS = load_file(Mname)     #there should only exist 1 model
            stepNum = i
            #classes = load_file('classes.pkl')
            for j in range(0,i+1):
                classes_learned = classes_learned + len(order[j][0])
                if (order[j][2] >= 0 and order[j][2] != frac_class):
                    classes_learned = classes_learned + 1
                    frac_class = order[j][2]
                #recreate dataset
                for k in order[j][0]:
                    datasets.sum_data(classes, k, 0)
                if order[j][1] > 0.0:
                    datasets.sum_data(classes, order[j][2], 0, percentage=order[j][1])

            break
    if(stepNum == 0):
        Mname = "model_"+"step_"+str(0)+".pkl"
        for k in initial_classes:
            datasets.sum_data(classes, k, 0)
        classes_learned = classes_learned + len(order[0][0])
        print("no existing model found")
        print("training step 0")
        print(datetime.datetime.now())
        print ("at step " + str(0) + " learn these " + str(order[0]) + " classes ")
        (score, RAS) = CNN_Training.train_Teacher(RAS, classes[0], teacher_architecture,
                output_dim=classes_learned, lr=lr, epochs=epochs, batch_size=batch_size, 
                verbose=False, patience=patience, test=True, class_curve=False, 
                save=False, data_augmentation=data_augmentation)
        print("[FTA] Teacher network achieved test accuracy " + str(score[1]))
        print("Teacher Network has " + str(RAS.numParams) + " parameters")
        save_object(RAS, Mname)
    else:
        print(Mname + " is found")


    '''
    initialize the search director 
    '''
    search_dir =  search_director.search_director(N_workers, lr, epoch_limit, discrete_levels,
                    patience=patience, data_augmentation=data_augmentation)
    stepNum = stepNum + 1
    checkpoint = True

    for i in range(stepNum, len(order)):
        print ("--------------------------------------------------------------")
        print("training step " + str(i))
        print(datetime.datetime.now())
        print ("saturate with new data")
        print ("at step " + str(i) + " learn these " + str(order[i]) + " classes ")
        for k in order[i][0]:
            datasets.sum_data(classes, k, 0)
        classes_learned = classes_learned + len(order[i][0])
        if order[i][1] > 0.0:
            classes = datasets.sum_data(classes, order[i][2], 0, percentage=order[i][1])
        if (order[i][2] >= 0 and order[i][2] != frac_class):
            classes_learned = classes_learned + 1
            frac_class = order[i][2]

        log_path = "./logs/"+str(i)+"_1RAS"
        
        (Vexist, RAS) = CNN_Training.execute(RAS, classes[0], output_dim=classes_learned,
                                    lr=lr, epochs=epochs, batch_size=batch_size,
                                    verbose=False, patience=patience,test=False, 
                                    log_path=log_path, baseline=False,
                                    upload=True, class_curve=False, save=False, data_augmentation=data_augmentation)
        print(datetime.datetime.now())
        params = RAS.export()
        print("[PSA]")
        RAS.print_arch()
        log_path = "./logs/"+str(i)+"_1RAS"
        print ("[PVA] before RAS search, model achieved validation accuracy of " + str(Vexist[1]))
        #obtain randomly sampled architectures 
        instructions = CNN_RAS.random_search(RAS.architecture, sample_size, wider_max_actions, deeper_max_actions, sample_size, fc_limit)
        (candidate, val_accus) = search_dir.search_step(instructions, params, Vexist[1], classes[0], classes_learned)
        print(datetime.datetime.now())

        #only use the candidate if it is better than existing architecture in val accu
        # if (candidate['accuracy'] > Vexist[1]):        
        #     RAS.param_import(candidate['params'])
        # else:
        #     print("RAS candidate is worse than existing architecture, keep existing architecture")
        RAS.param_import(candidate['params'])
        preTrainedVal = candidate['accuracy']

        print("[PSA]")
        RAS.print_arch()
        print("RAS candidate achieved validation acccuracy of " + str(candidate['accuracy']) + " before further training ")

        (predictions, pre_score) = CNN_Training.class_evaluation(RAS, classes[0], output_dim=classes_learned,
                                                                lr=lr, batch_size=batch_size, verbose=False)
        print("RAS candidate achieved test accuracy of " + str(pre_score[1]) + " before further training ")

        #further training for RAS
        RAS_copy = copy.deepcopy(RAS)       #in case we need to revert back after further training
        (score, RAS) = CNN_Training.execute(RAS, classes[0], output_dim=classes_learned,
                                    lr=lr, epochs=epochs, batch_size=batch_size,
                                    verbose=False, patience=patience,test=True, 
                                    log_path=log_path, baseline=False,
                                    upload=True, class_curve=False, save=False, data_augmentation=data_augmentation)
        TrainedVal = CNN_Training.get_val_accu(RAS, classes[0], output_dim=classes_learned, lr=lr, batch_size=batch_size, verbose=False)

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

        fname = "model_"+"step_"+str(i)+".pkl"
        save_object(RAS, fname)
        fdelete = "model_"+"step_"+str(i-1)+".pkl"
        delete_file(fdelete)

        print(datetime.datetime.now())
        print("training step " + str(i) + " completed")
        print("--------------------------------------------------------------")


def main():
    learning()



if __name__ == "__main__":
    main()
