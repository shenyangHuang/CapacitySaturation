# For reproducibility of the experiments
# However, gpu might still generate stochasticity
seed_value = 5


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
from Dataset_Management import datasets, memory
from Model import CNN_Model
from Params import arch_templates
from Params.param_templates import p_s
from Model.Layer_obj import Layer_obj
from Training import CNN_Training
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
conduct 2-class incremental learning on CIFAR 100 dataset with a fixed architecture
'''

def learning():
    FTA_scores=[]
    FTA_time = []

    lr = p_s["lr"]
    epochs = p_s["epochs"]
    patience = p_s["patience"]
    batch_size = 16

    mem_size = 2000
    order = list(range(100)) 
    classes = datasets.Incremental_partition("cifar100", 0.034, order, normalization="numerical")
    fixedCNN = CNN_Model.CNN_Model("fixed")
    teacher_architecture = arch_templates.CNAS_start_arch()
    memoryBuffer = memory.ResevoirSampling(mem_size)
    valMemory = classes[0]

    #first check if there is existing model
    initNum = 10         #2 base knowledge class
    startNum = initNum      #can resume from any number of classes 
    increment = 2       #add in 2 new classes at a time
    Mname = "model"+str(initNum)+".pkl"

    for i in range(initNum,100):
        Mname = "model"+str(i)+".pkl"
        if(os.path.isfile(Mname)):
            fixedCNN = load_file(Mname)     #there should only exist 1 model
            startNum = i
            for k in range(1, i):
                datasets.sum_data(classes, k, 0, percentage=1.0) 
            break
    if(startNum == initNum):
        for k in range(1,startNum):
            datasets.sum_data(classes, k, 0, percentage=1.0)            #use first 2 classes as base knowledge
        print("no existing model found")
        print("training step 0")
        print(process_time())
        (score, fixedCNN) = CNN_Training.train_Teacher(fixedCNN, classes[0], teacher_architecture,
                output_dim=startNum, lr=lr, epochs=epochs, 
                batch_size=batch_size, verbose=False, patience=patience, 
                test=True, class_curve=False, save=False, data_augmentation=False)
        print("[FTA] Teacher network achieved test accuracy " + str(score[1]))
        FTA_scores.append(score[1])
        FTA_time.append(process_time())
        memoryBuffer.update(classes[0])
        save_object(fixedCNN, "model"+str(startNum)+".pkl")
    else:
        print(Mname + " is found")

    (predictions, pre_score) = CNN_Training.class_evaluation(fixedCNN, classes[0], output_dim=startNum, lr=lr, batch_size=batch_size, verbose=False)

    checkpoint = True
    for i in range(int(startNum/increment),int(100/increment)):
        print ("--------------------------------------------------------------")
        print("training step " + str((i+1)*increment))

        #adding new validation data / update validation memory
        for k in range(i*increment, (i+1)*increment):
            datasets.sum_data(classes, k, 0, percentage=1.0)        #add 2 new classes at a time

        #new training data
        for k in range(i*increment+1, (i+1)*increment):
            datasets.sum_data(classes, k, i*increment, percentage=1.0)


        log_path = "./logs/"+str((i+1)*increment)+"_fixed"

        (score, fixedCNN) = CNN_Training.episodic_train(fixedCNN, classes[i*increment], valMemory, memoryBuffer, output_dim=((i+1)*increment),
                                                        lr=lr, batch_size=batch_size, epochs=epochs, patience=patience,
                                                        verbose=False, test=True, log_path=log_path, 
                                                        baseline=False, upload=True, class_curve=False, 
                                                        save=False, data_augmentation=False, tensorBoard=False)

        print("[ICA]")
        (predictions, pre_score) = CNN_Training.class_evaluation(fixedCNN, classes[0], output_dim=((i+1)*increment), lr=lr, batch_size=batch_size, verbose=False)
        print("[FTA] network achieved test accuracy " + str(pre_score[1]) + " at step " + str((i+1)*increment))
        FTA_scores.append(pre_score[1])
        FTA_time.append(process_time())
        memoryBuffer.update(classes[i*increment])
        K.clear_session()
        print ("all test accuracies:")
        print (FTA_scores)
        print ("all test time is:")
        print (FTA_time)


            
def main():
    learning()

if __name__ == "__main__":
    main()







