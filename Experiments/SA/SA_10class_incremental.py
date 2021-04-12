# For reproducibility of the experiments
# However, gpu might still generate stochasticity
seed_value = 3

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
from Params.param_templates import p_s
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

    lr = p_s["lr"]
    epochs = p_s["epochs"]
    patience = p_s["patience"]
    batch_size = p_s["batch_size"]

    print ("-------------------------------------")
    print ("this experiment transfer weights")
    print ("-------------------------------------")

    FTA_scores = []
    order = list(range(100)) 
    classes = datasets.Incremental_partition("cifar100", 0.1, order, normalization="numerical")
    fixedCNN = CNN_Model.CNN_Model("fixed")

    
    teacher_architecture = arch_templates.SA_10class()

    #first check if there is existing model
    Mname = "model"+str(10)+".pkl"
    startNum = 10
    for i in range(10,100):
        Mname = "model"+str(i)+".pkl"
        if(os.path.isfile(Mname)):
            fixedCNN = load_file(Mname)     #there should only exist 1 model
            startNum = i
            for k in range(1, i):
                datasets.sum_data(classes, k, 0, percentage=1.0) 
            break
    if(startNum == 10):
        for k in range(1,10):
            datasets.sum_data(classes, k, 0, percentage=1.0)            #use first 10 classes as base knowledge
        print("no existing model found")
        print("training step 0")
        print(datetime.datetime.now())
        (score, fixedCNN) = CNN_Training.train_Teacher(fixedCNN, classes[0], teacher_architecture,
                output_dim=startNum, lr=lr, epochs=epochs, 
                batch_size=batch_size, verbose=False, patience=patience, 
                test=True, class_curve=False, save=False, data_augmentation=False)
        print("[FTA] Teacher network achieved test accuracy " + str(score[1]))
        FTA_scores.append(score[1])
        save_object(fixedCNN, "model"+str(10)+".pkl")
    else:
        print(Mname + " is found")

    (predictions, pre_score) = CNN_Training.class_evaluation(fixedCNN, classes[0], output_dim=startNum, lr=lr, batch_size=batch_size, verbose=False)

    checkpoint = True
    for i in range(int(startNum/10),10):
        print ("--------------------------------------------------------------")
        print("training step " + str((i+1)*10))
        print(datetime.datetime.now())
        for k in range(i*10, (i+1)*10):
            datasets.sum_data(classes, k, 0, percentage=1.0)        #add 10 new classes at a time
        log_path = "./logs/"+str((i+1)*10)+"_fixed"

        (score, fixedCNN) = CNN_Training.execute(fixedCNN, classes[0], output_dim=((i+1)*10),
                lr=lr, epochs=epochs, batch_size=batch_size,
                verbose=False, patience=patience, test=True, 
                log_path=log_path, baseline=False,
                upload=True, class_curve=False, save=False, data_augmentation=False)
        print(datetime.datetime.now())
        print("[FTA] network achieved test accuracy " + str(score[1]) + " at step " + str((i+1)*10))
        FTA_scores.append(score[1])
        print("[ICA]")
        (predictions, pre_score) = CNN_Training.class_evaluation(fixedCNN, classes[0], output_dim=((i+1)*10), lr=lr, batch_size=batch_size, verbose=False)
        pname = "predictions" + str((i+1)*10) + ".pkl" 

        K.clear_session()
        print ("all test accuracies are: ")
        print (FTA_scores)


        
def main():
    learning()

if __name__ == "__main__":
    main()







