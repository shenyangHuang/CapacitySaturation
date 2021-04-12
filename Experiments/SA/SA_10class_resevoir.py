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
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 5. limit gpu usage and doesn't take the entire gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))



from time import process_time
from Dataset_Management import datasets, memory
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
    #let it be 10 samples per class in the memory
    FTA_time = []
    mem_size = 1000

    order = list(range(100)) 
    #only keep a small percent of samples in the validation set
    classes = datasets.Incremental_partition("cifar100", 0.017, order, normalization="numerical")



    FTA_scores=[]

    lr = p_s["lr"]
    patience = p_s["patience"]
    batch_size = p_s["batch_size"]
    patience = p_s["patience"]
    epochs = p_s["epochs"]
    batch_size = 16


    fixedCNN = CNN_Model.CNN_Model("fixed")
    teacher_architecture = arch_templates.SA_10class()
    memoryBuffer = memory.ResevoirSampling(mem_size)
    valMemory = classes[0]


    #first check if there is existing model
    Mname = "model"+str(10)+".pkl"
    startNum = 10
    for i in range(10,100):
        Mname = "model"+str(i)+".pkl"
        if(os.path.isfile(Mname)):
            fixedCNN = load_file(Mname)     #there should only exist 1 model
            startNum = i
            break
    if(startNum == 10):
        for k in range(1,10):
            datasets.sum_data(classes, k, 0, percentage=1.0)            #use first 10 classes as base knowledge

        print("no existing model found")
        print("training step 0")
        print(process_time())
        
        (score, fixedCNN) = CNN_Training.train_Teacher(fixedCNN, classes[0], teacher_architecture,
                output_dim=startNum, lr=lr, epochs=epochs, 
                batch_size=batch_size, verbose=False, patience=patience, 
                test=True, class_curve=False, save=False, data_augmentation=False)
        print("[FTA] Teacher network achieved test accuracy : " + str(score[1]))
        memoryBuffer.update(classes[0])
        FTA_scores.append(score[1])
        FTA_time.append(process_time())
        #save_object(fixedCNN, "model"+str(10)+".pkl")
    else:
        print(Mname + " is found")
    
    (predictions, pre_score) = CNN_Training.class_evaluation(fixedCNN, classes[0], output_dim=startNum, lr=lr, batch_size=batch_size, verbose=False)

    checkpoint = True
    for i in range(int(startNum/10),10):
        print ("--------------------------------------------------------------")
        print("training step " + str((i+1)*10))
        print(process_time())
        '''
        seeing 10 new classes
        sum it to position i*10
        '''
        datasets.sum_data(classes, i*10, 0, percentage=1.0)
        for k in range(i*10 + 1, (i+1)*10):
            datasets.sum_data(classes, k, i*10, percentage=1.0)        #add 10 new classes at a time
            datasets.sum_data(classes, k, 0, percentage=1.0)   #for testing purpose
        
        log_path = "./logs/"+str((i+1)*10)+"_fixed"
        (score, fixedCNN) = CNN_Training.episodic_train(fixedCNN, classes[i*10], valMemory, memoryBuffer, output_dim=((i+1)*10),
                                                        lr=lr, batch_size=batch_size, epochs=epochs, patience=patience,
                                                        verbose=False, test=True, log_path="none", 
                                                        baseline=False, upload=True, class_curve=False, 
                                                        save=False, data_augmentation=False, tensorBoard=False)
        memoryBuffer.update(classes[i*10])
        FTA_time.append(process_time())
        print("[ICA]")
        (predictions, pre_score) = CNN_Training.class_evaluation(fixedCNN, classes[0], output_dim=((i+1)*10), lr=lr, batch_size=batch_size, verbose=False)
        print("[FTA] network achieved test accuracy " + str(pre_score[1]) + " at step " + str((i+1)*10))
        FTA_scores.append(pre_score[1])

        K.clear_session()
        print ("all test scores")
        print (FTA_scores)
        print ("-----------------------------")
        print ("all compute times are")
        print (FTA_time)

        
def main():
    learning()

if __name__ == "__main__":
    main()







