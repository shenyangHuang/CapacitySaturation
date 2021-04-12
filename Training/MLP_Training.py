'''
Use training functions for CNN here
'''
from __future__ import print_function
from .per_class import class_accuracy
from .count_epoch import count_epoch
from Transformations import MLP_transformation
from Model import MLP_Model
from Model.Layer_obj import Layer_obj
from Training import Curriculum
from Dataset_Management import datasets
import numpy as np
import copy
import pickle
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks
from keras import backend as K

#uses Adam as optimizer
#now architecture store weight shapes instead of neuron number

'''
build and compile a model
'''
def build_model(mlp_model, architecture, input_dim, output_dim):
    
    model = Sequential()

    #first layer is always conv :)
    fc = architecture[0].size
    model.add(Dense(fc[1], activation='relu', input_dim=input_dim))

    for i in range(1,len(architecture)):

        #all types should be fc
        if(architecture[i].Ltype == 'fc'):
            fc = architecture[i].size
            model.add(Dense(fc[1], activation='relu'))
        else:
            continue

    model.add(Dense(output_dim, activation='softmax'))
    return model


def compile_model(mlp_model, model, lr, filepath, verbose,
                patience, log_path, log=False, 
                class_curve=True, tensorBoard=False, teacher=False):

    count = 0
    if (not teacher):
        for i in range(0,len(model.layers)):
            if(not model.layers[i].get_weights()): 
                continue
            if(len(mlp_model.weights[count]) > 1):
                model.layers[i].set_weights(mlp_model.weights[count])       #normal layer with weights and bias
            count = count + 1
    adam = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
          optimizer=adam,
          metrics=['accuracy'])

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=verbose, save_best_only=True, mode='auto')
    num_epochs = count_epoch()
    CallList = [checkpoint, num_epochs]
    if(patience>0):
        early_Stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=verbose, mode='auto')
        CallList.append(early_Stopping)
    if(log):
        logCall = callbacks.CSVLogger(log_path, separator=',', append=False)
        CallList.append(logCall)
    if(class_curve):
        class_accu = class_accuracy()
        CallList.append(class_accu)
    if(tensorBoard):
        boardCall = callbacks.TensorBoard('./Graph',  histogram_freq=0, write_graph=True, write_images=True)
        CallList.append(boardCall)
    return (model,CallList)

'''
build and compile a model
train a model
'''
def train_model(model, x_train, y_train, x_val, y_val, epochs=100, 
                batch_size=128, verbose=False, callbacks=None, 
                shuffle=True):

    model.fit(x_train, y_train, epochs=epochs, 
            batch_size=batch_size, verbose=verbose, callbacks=callbacks,  
            validation_data=(x_val,y_val), shuffle=shuffle)
    return model


'''
Train a simple teacher network
must take an architecture blueprint as input
'''
def train_Teacher(mlp_model, dataset, architecture,
                output_dim=10, lr=0.0001, epochs=100, 
                batch_size=128, verbose=False, patience=20, 
                test=True, class_curve=False, save=False, 
                tensorBoard=False):

    '''
    preparing training, validation and test set
    '''
    if type(dataset.train_images) is not np.ndarray:
        raise ValueError('The dataset is not in numpy ndarray format')
    x_train, y_train = dataset.train_images, dataset.train_labels
    x_val, y_val = dataset.validation_images, dataset.validation_labels
    x_test, y_test = dataset.test_images, dataset.test_labels

    #for MNIST and Fashion MNIST
    input_dim = 784
    
    '''
    build the model
    '''
    filepath=mlp_model.name + "teacher.best.hdf5"
    model = build_model(mlp_model, architecture, input_dim, output_dim)

    '''
    compile the model
    '''
    model, CallList = compile_model(mlp_model, model, lr, filepath, verbose,
                patience, "none", log=False, class_curve=class_curve, tensorBoard=tensorBoard, teacher=True)
    '''
    Training happens here
    '''
    model = train_model(model, x_train, y_train, x_val, y_val, epochs=epochs, 
        batch_size=batch_size, verbose=verbose, callbacks=CallList, shuffle=True)

    if (epochs > 0):
        model.load_weights(filepath)
    

    score = 0
    if(test):
        score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    else:
        score = model.evaluate(x_val, y_val, batch_size=batch_size,verbose=0)
    
    weights = []

    #upload the weights 
    for i in range(0,len(model.layers)):
        parameters = model.layers[i].get_weights()
        #layers doesn't have weight 
        if(not parameters):
            continue
        if(len(parameters) > 2):
            weights.append([parameters])
        else:
            weights.append(parameters)
    mlp_model.weights = weights
    mlp_model.architecture = architecture
    mlp_model.numParams = model.count_params()


    if(save):
        #model.save(mlp_model.name + 'teacher.h5')
        save_object(mlp_model, mlp_model.name + 'teacher.pkl')
        print (mlp_model.name + 'teacher.pkl' + 'saved to disk')

    os.remove(filepath)       #remove the stored weight file 
    return (score,mlp_model)


'''
Train a transformed MLP
'''
def execute(mlp_model, dataset, output_dim=10,
            lr=0.001, epochs=100, batch_size=32,
            verbose=False, patience=20, test=True,
            log_path="none", upload=True, class_curve=False, 
            save=False, tensorBoard=False):

    log = False
    if(log_path != "none"):
        log = True

    past_dim = len(mlp_model.weights[-1][1])
    if(output_dim > past_dim):
        mlp_model = MLP_transformation.outLayer_transform(mlp_model, output_dim)
    if type(dataset.train_images) is not np.ndarray:
        raise ValueError('The dataset is not in numpy ndarray format')
    architecture = mlp_model.architecture

    x_train, y_train = dataset.train_images, dataset.train_labels
    x_val, y_val = dataset.validation_images, dataset.validation_labels
    x_test, y_test = dataset.test_images, dataset.test_labels

    
    #For MNIST
    input_dim = 784

    '''
    Build the model 
    '''
    filepath=mlp_model.name + "execute.best.hdf5"
    model = build_model(mlp_model, architecture, input_dim, output_dim)

    '''
    Compile the Model
    '''
    model = build_model(mlp_model, architecture, input_dim, output_dim)
    model, CallList = compile_model(mlp_model, model, lr, filepath, verbose,
                patience, log_path, log=log, class_curve=class_curve, tensorBoard=tensorBoard)

    '''
    training process 
    '''
    model = train_model(model, x_train, y_train, x_val, y_val, epochs=epochs, 
                batch_size=batch_size, verbose=verbose, callbacks=CallList, 
                shuffle=True)

    if (epochs > 0):
        model.load_weights(filepath)    
    score = 0
    if(test):
        score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    else:
        score = model.evaluate(x_val, y_val, batch_size=batch_size,verbose=0)

    mlp_model.numParams = model.count_params()

    if(upload):
        weights = []
        for i in range(0,len(model.layers)):
            parameters = model.layers[i].get_weights()
            if(not parameters):
                continue
            if (len(parameters) > 2):
                weights.append([parameters])
            else:
                weights.append(parameters)
        mlp_model.weights = weights
    if(save):
        #model.save(mlp_model.name + str(output_dim) + '.h5')
        save_object(mlp_model, mlp_model.name + str(output_dim) + '.pkl')
        print (mlp_model.name + str(output_dim) + '.pkl' + 'saved to disk')

    os.remove(filepath)       #remove the stored weight file 
    return (score,mlp_model)

'''
returns the predictions made
'''

def class_evaluation(mlp_model, dataset,
    output_dim=10, lr=0.001, batch_size=32,
    verbose=False):

    # self.gpu_limit()
    past_dim = len(mlp_model.weights[-1][1])
    if(output_dim > past_dim):
        mlp_model = MLP_transformation.outLayer_transform(mlp_model, output_dim, avg=False)
    x_test,y_test = dataset.test_images, dataset.test_labels
    architecture = mlp_model.architecture
    #For MNIST
    input_dim = 784

    filepath = "not_really.path"
    model = build_model(mlp_model, architecture, input_dim, output_dim)
    model, CallList = compile_model(mlp_model, model, lr, filepath, verbose,
                20, "none", log=False, class_curve=False)

    predictions = model.predict(x_test,verbose=verbose)
    class_count = np.zeros((output_dim,2))
    for i in range(0,len(predictions)):
        correct = np.argmax(y_test[i])
        class_count[correct][1] = class_count[correct][1] + 1
        if( (np.argmax(predictions[i])) == correct):
            class_count[correct][0] = class_count[correct][0] + 1

    for i in range(0,output_dim):
        print ("class " + str(i+1) + " : " + str(class_count[i][0]) + " / " + str(class_count[i][1]))
        accuracy = float(class_count[i][0]) / float(class_count[i][1])
        print ("test accuracy is " + str(accuracy) )

    score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    print ("overall average test accuracy is " + str(score[1]))
    return (predictions,score)

def get_val_accu(mlp_model, dataset,
    output_dim=10, lr=0.001, batch_size=32,
    verbose=False):

    past_dim = len(mlp_model.weights[-1][1])
    if(output_dim > past_dim):
        mlp_model = MLP_transformation.outLayer_transform(mlp_model, output_dim, avg=False)
    x_val,y_val = dataset.validation_images, dataset.validation_labels
    architecture = mlp_model.architecture
    #for MNIST
    input_dim = 784

    filepath = "not_really.path"
    model = build_model(mlp_model, architecture, input_dim, output_dim)
    model, CallList = compile_model(mlp_model, model, lr, filepath, verbose,
                20, "none", log=False, class_curve=False)

    score = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
    return score[1]
    

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    return pickle.load( open( filename, "rb" ) )



















































