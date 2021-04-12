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
from Dataset_Management import datasets
from Model import CNN_Model
from Model.Layer_obj import Layer_obj
from Params import arch_templates
from Transformations import CNN_transformation
from Training import CNN_Training, Curriculum
from Measurements.KnaryClassification import knaryAccuracy
import os
import pickle
import numpy as np

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def delete_file(filename):
    if(os.path.isfile(filename)):
        os.remove(filename)

def load_file(filename):
    return pickle.load( open( filename, "rb" ) )



'''
conduct 10-class incremental learning on CIFAR 100 dataset with a fixed architecture
'''

def learning():

    lr = 0.0001
    epochs = 1000
    patience = 10
    batch_size = 128
    minRange = 2    #minimum number of new classes arrive at each step
    maxRange = 19   #maximum number of new classes arrive at each step 
    total = 100     #total number of classes in cifar100


    order = list(range(100)) 
    classes = datasets.Incremental_partition("cifar100", 0.1, order, normalization="numerical")
    print ("integer order")
    if(os.path.isfile("ordering.pkl")):
        order = load_file("ordering.pkl")

    else:
        order = datasets.variable_K_integer(total, minRange, maxRange)
        save_object(order, "ordering.pkl")



    fixedCNN = CNN_Model.CNN_Model("fixed")

    teacher_architecture = arch_templates.static_arch()


    #first check if there is existing model
    Mname = "model_"+"step_"+str(0)+".pkl"
    stepNum = 0
    classes_learned = 0
    for i in range(0,100):
        Mname = "model_"+"step_"+str(i)+".pkl"
        if(os.path.isfile(Mname)):
            fixedCNN = load_file(Mname)     #there should only exist 1 model
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
        (score, fixedCNN) = CNN_Training.train_Teacher(fixedCNN, classes[0], teacher_architecture,
                output_dim=classes_learned, lr=lr, epochs=epochs, 
                batch_size=batch_size, verbose=False, patience=patience, 
                test=True, class_curve=False, save=False, data_augmentation=False)
        print("[FTA] Teacher network achieved test accuracy " + str(score[1]))
        print("Teacher Network has " + str(fixedCNN.numParams) + " parameters")
        save_object(fixedCNN, Mname)
    else:
        print(Mname + " is found")

    (predictions, pre_score) = CNN_Training.class_evaluation(fixedCNN, classes[0], output_dim=classes_learned, lr=lr, batch_size=batch_size, verbose=False)
    stepNum = stepNum + 1
    checkpoint = True
    for i in range(stepNum,len(order)):
        print ("--------------------------------------------------------------")
        print("training step " + str(i))
        print(datetime.datetime.now())
        for k in order[i]:
            datasets.sum_data(classes, k, 0)
        print ("at step " + str(i) + " learn these " + str(order[i]) + " classes ")
        classes_learned = classes_learned + len(order[i])
        log_path = "./logs/"+"step_"+ str(i)+"_fixed"

        (score, fixedCNN) = CNN_Training.execute(fixedCNN, classes[0], output_dim=classes_learned,
                lr=lr, epochs=epochs, batch_size=batch_size,
                verbose=False, patience=patience, test=True, 
                log_path=log_path, baseline=False,
                upload=True, class_curve=False, save=False, data_augmentation=False)
        print(datetime.datetime.now())
        print("[FTA] network achieved test accuracy " + str(score[1]) + " at step " + str(i))
        print("[ICA]")
        (predictions, pre_score) = CNN_Training.class_evaluation(fixedCNN, classes[0], output_dim=classes_learned, lr=lr, batch_size=batch_size, verbose=False)
        pname = "predictions" + "step_"+ str(i) + ".pkl" 
        
        #save_object(predictions, pname)     #save predictions
        fname = "model_"+"step_"+ str(i)+".pkl"
        save_object(fixedCNN, fname)
        fdelete = "model_"+"step_"+ str(i-1)+".pkl"
        delete_file(fdelete)

        K.clear_session()

        
def main():
    learning()

if __name__ == "__main__":
    main()







