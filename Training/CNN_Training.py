'''
Use training functions for CNN here
'''
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from .per_class import class_accuracy
from .count_epoch import count_epoch
from Transformations import CNN_transformation
from Model import CNN_Model
from Model.Layer_obj import Layer_obj
from Training import Curriculum
from Dataset_Management import datasets, memory
import numpy as np
import copy
import pickle
import os
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks
from keras import backend as K

#uses Adam as optimizer
#now architecture store weight shapes instead of neuron number
#architecture will now store batch normalization information as well

'''
build and compile a model
'''
def build_model(cnn_model, architecture, input_shape, output_dim):
    
    model = Sequential()

    #first layer is always conv :)
    conv = architecture[0].size
    model.add(Conv2D(conv[-1], (conv[0], conv[1]), 
                    padding='same',activation='relu',
                    input_shape=input_shape))
    
    #used to indicate if the first dense layer has been added 
    #must flatten the input before first dense layer
    firstDense = True

    for i in range(1,len(architecture)):

        if(architecture[i].Ltype == 'conv'):
            conv = architecture[i].size
            model.add(Conv2D(conv[-1], (conv[0], conv[1]), 
                    padding='same',activation='relu'))           
        
        elif(architecture[i].Ltype == 'maxpool'):
            pool = architecture[i].size
            model.add(MaxPooling2D(pool_size=(pool[0],pool[0]), strides=(2, 2)))
        
        elif(architecture[i].Ltype == 'bn'):
            model.add(BatchNormalization())

        elif(architecture[i].Ltype == 'fc'):        #dense layer 
            if(firstDense):
                model.add(Flatten())
                firstDense = False
            fc = architecture[i].size
            model.add(Dense(fc[1], activation='relu'))

        elif(architecture[i].Ltype == 'dropout'):
            dropout = architecture[i].size
            model.add(Dropout(dropout[0]))

        else:
            continue

    model.add(Dense(output_dim, activation='softmax'))
    # model.summary()
    return model


def compile_model(cnn_model, model, lr, filepath, verbose,
                patience, log_path, log=False, 
                class_curve=True, baseline=False, tensorBoard=False):

    if (not baseline):
        count = 0
        for i in range(0,len(model.layers)):
            if(not model.layers[i].get_weights()):      #dropout and maxpool layer doesn't have weights
                continue
            #newly added BN layer will have len = 0 so it is not processed here
            if(len(cnn_model.weights[count]) == 1):
                model.layers[i].set_weights(cnn_model.weights[count][0])    #BN layer 
            if(len(cnn_model.weights[count]) > 1):
                model.layers[i].set_weights(cnn_model.weights[count])       #normal layer with weights and bias
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
                batch_size=32, verbose=False, callbacks=None, 
                shuffle=True, data_augmentation=True):

    if not data_augmentation:
        model.fit(x_train, y_train, epochs=epochs, 
                batch_size=batch_size, verbose=verbose, callbacks=callbacks,  
                validation_data=(x_val,y_val), shuffle=shuffle)

    # This will do preprocessing and realtime data augmentation:
    else:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=(len(x_train) / batch_size), verbose=verbose, validation_data=(x_val, y_val),
        epochs=epochs, callbacks=callbacks, shuffle=shuffle)

    return model

'''
training the model with various sampling techniques from the memory, 
explicitly control what is in each batch 
uses keras train_on_batch function
https://keras.io/models/model/

x_train, y_train are the data of the new classes arrivings
x_val, y_val are from the validation memory
'''
def batch_training(model, x_train, y_train, x_val, y_val, epochs, patience, memoryBuffer, filepath, batch_size=32):
    #shuffle the samples
    samples = []
    for i in range(len(x_train)):
        samples.append((x_train[i], y_train[i]))
    random.shuffle(samples)

    print ("learning new class " + str(y_train[0]))

    min_loss = 10000000000
    stopped_epochs = 0
    improved = False

    num_batch = int(len(samples)/batch_size) + (len(samples) % batch_size > 0)
    epos = 0
    for z in range(epochs):
        improved = False
        idx = 0
        for k in range(num_batch):
            x = []
            y = []
            #half are new samples
            #half are randomly picked from memory

            ms = memoryBuffer.minibatch(int(batch_size/2), y_train[0].shape[-1])
            remainder = batch_size - int(batch_size/2)

            for i in range(remainder):
                x.append(samples[idx][0])
                y.append(samples[idx][1])
                idx = idx + 1

            for i in range(int(batch_size/2)):
                x.append(ms[i][0])
                y.append(ms[i][1])


            x = np.asarray(x)
            y = np.asarray(y)
            losses = model.train_on_batch(x, y)
            loss = sum(losses) / len(losses)
            if (loss < min_loss):
                min_loss = loss
                model.save_weights(filepath)
                improved = True

        if (not improved):
            stopped_epochs = stopped_epochs + 1
        else:
            stopped_epochs = 0
        if (stopped_epochs >= patience):
            break

        epos = epos + 1

    print ("training used " + str(epos) + " epochs")

    return model







'''
Gather statistics from a forward pass to undo the normalization for a newly added BN layer
tasks indicates the layers that we need to get statistics from 
'''

# def update_statistics(cnn_model, dataset, output_dim, 
#                     batch_size, noVal, data_augmentation):
    
#     architecture = cnn_model.architecture
#     TaskList = cnn_model.tasks
#     print ("Tasks for updating statistics are")
#     for task in TaskList:
#         print (task)

#     x_train, y_train = dataset.train_images, dataset.train_labels
#     x_val, y_val = dataset.validation_images, dataset.validation_labels
#     x_test, y_test = dataset.test_images, dataset.test_labels
#     #concatenate validation data with training data if needed
#     if(noVal):
#         x_train = np.concatenate((x_train,x_val), axis=0)
#         y_train = np.concatenate((y_train,y_val), axis=0)
#     input_shape = x_train.shape[1:]

#     '''
#     build and compile the model
#     '''
#     model = build_model(cnn_model, architecture, input_shape, output_dim)
#     model, CallList = compile_model(cnn_model, model, 0.0001, "none", False,
#                 20, "none", log=False, class_curve=False, baseline=False, tensorBoard=False)

#     for task in cnn_model.tasks:
#         print ("ready to update layer " + str(task))


#     inp = model.input     # input placeholder
#     print ("gather statistics on layer " + str(task-1))
#     taskLayers = [model.get_layer(index=task-1) for task in TaskList]  #select the newly added identity layer
#     outputs = [layer.output for layer in taskLayers]          # tasked layer outputs
#     functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions


#     x_batches = []
#     y_batches = []  #don't really need y_batch but still put here to use with generator
#     if (data_augmentation):    #use data augmentation for each batch
#         datagen = ImageDataGenerator(
#             featurewise_center=False,  # set input mean to 0 over the dataset
#             samplewise_center=False,  # set each sample mean to 0
#             featurewise_std_normalization=False,  # divide inputs by std of the dataset
#             samplewise_std_normalization=False,  # divide each input by its std
#             zca_whitening=False,  # apply ZCA whitening
#             rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#             width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#             height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#             horizontal_flip=True,  # randomly flip images
#             vertical_flip=False)  # randomly flip images
#         datagen.fit(x_train)
#         for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
#             x_batches.append(x_batch)

#     else:
#         for cbatch in range(0, x_train.shape[0], batch_size):
#             x_batches.append(x_train[cbatch:(cbatch + batch_size),:,:])

#     batch_num = len(x_batches)
#     statistics = [[0, 0] for _ in TaskList]

#     for _i in range(0,batch_num):
#         layer_outs = [func([x_batches[_i]]) for func in functors] #one batch at a time
#         layer_outs = np.asarray(layer_outs)       #get the activations for this batch
#         for _j, out in enumerate(outputs):
#             layer_outs = layer_outs.astype('float32')
#             axis = tuple(range(len(layer_outs.shape) - 1))
#             mean = np.mean(layer_outs, axis=axis, keepdims=True)
#             variance = np.mean(np.square(layer_outs - mean), axis=axis, keepdims=True)
#             mean, variance = np.squeeze(mean), np.squeeze(variance)
#             statistics[_j][0] += mean
#             statistics[_j][1] += variance

#     #find empty lists within architecture
#     archIdx = []
#     for _i in range(0, len(architecture)):
#         #unfinished task
#         if (len(architecture[_i].size) == 0):
#             archIdx.append(_i)

#     for _k in range(0,len(statistics)):
#         mean, variance = statistics[_k][0] / batch_num, statistics[_k][1] / batch_num   #moving mean and moving variance
#         BN_params = CNN_transformation.set_BN_identity(statistics[_k])
#         num_hidden = len(BN_params[0])
#         cnn_model.weights[TaskList[_k]] = [BN_params]     #update weight list
#         #update architecture list
#         cnn_model.architecture[archIdx[_k]].size = [num_hidden]

#     cnn_model.tasks = []        #task completed
#     return cnn_model



'''
Train a simple teacher network
must take an architecture as input
'''
def train_Teacher(cnn_model, dataset, architecture,
                output_dim=10, lr=0.001, epochs=20, batch_size=32, verbose=False, 
                patience=20, test=True, class_curve=False, save=False, 
                data_augmentation=True, tensorBoard=False):

    # self.gpu_limit()
    '''
    preparing training, validation and test set
    '''
    if type(dataset.train_images) is not np.ndarray:
        raise ValueError('The dataset is not in numpy ndarray format')
    x_train, y_train = dataset.train_images, dataset.train_labels
    x_val, y_val = dataset.validation_images, dataset.validation_labels
    x_test, y_test = dataset.test_images, dataset.test_labels
    input_shape = x_train.shape[1:]
    
    '''
    build the model
    '''
    filepath=cnn_model.name + "teacher.best.hdf5"
    model = build_model(cnn_model, architecture, input_shape, output_dim)

    '''
    compile the model
    '''
    model, CallList = compile_model(cnn_model, model, lr, filepath, verbose,
                patience, "none", log=False, class_curve=class_curve, baseline=True, tensorBoard=tensorBoard)
    '''
    Training happens here
    '''
    model = train_model(model, x_train, y_train, x_val, y_val, epochs=epochs, 
            batch_size=batch_size, verbose=verbose, callbacks=CallList, 
            shuffle=True, data_augmentation=data_augmentation) 

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
    cnn_model.weights = weights
    cnn_model.architecture = architecture
    cnn_model.numParams = model.count_params()
    # model.summary()
    if(save):
        #model.save(cnn_model.name + 'teacher.h5')
        save_object(cnn_model, cnn_model.name + 'teacher.pkl')
        print (cnn_model.name + 'teacher.pkl' + 'saved to disk')

    os.remove(filepath)       #remove the stored weight file 
    return (score,cnn_model)



'''
Train with episodic memory
'''
def episodic_train(cnn_model, new_dataset, valMemory, memoryBuffer, output_dim=10,
            lr=0.001, batch_size=32, epochs=20, patience=20,
            verbose=False, test=True, log_path="none", 
            baseline=False, upload=True, class_curve=False, 
            save=False, data_augmentation=True, tensorBoard=False):
    log = False
    if(log_path != "none"):
        log = True

    past_dim = len(cnn_model.weights[-1][1])
    if(output_dim > past_dim):
        cnn_model = CNN_transformation.outLayer_transform(cnn_model, output_dim, avg=False)
    if type(new_dataset.train_images) is not np.ndarray:
        raise ValueError('The dataset is not in numpy ndarray format')
    architecture = cnn_model.architecture

    x_train, y_train = new_dataset.train_images, new_dataset.train_labels
    x_val, y_val = valMemory.validation_images, valMemory.validation_labels
    x_test, y_test = valMemory.test_images, valMemory.test_labels

    input_shape = x_train.shape[1:]

    '''
    Build the model 
    '''
    filepath=cnn_model.name + "execute.best.hdf5"
    model = build_model(cnn_model, architecture, input_shape, output_dim)


    '''
    finish the tasks for all newly added identity layer BN 
    '''
    if(len(cnn_model.tasks) > 0):
        raise ValueError('please Update all newly added BN before execution')

    '''
    Compile the Model
    '''
    model = build_model(cnn_model, architecture, input_shape, output_dim)
    model, CallList = compile_model(cnn_model, model, lr, filepath, verbose,
                patience, log_path, log=log, class_curve=class_curve, baseline=baseline, tensorBoard=tensorBoard)

    '''
    training process 
    '''
    model = batch_training(model, x_train, y_train, x_val, y_val, epochs, patience, memoryBuffer, filepath, batch_size=batch_size)

    if (epochs > 0):
        model.load_weights(filepath)

    score = 0
    if(test):
        score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    else:
        score = model.evaluate(x_val, y_val, batch_size=batch_size,verbose=0)

    cnn_model.numParams = model.count_params()

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
        cnn_model.weights = weights

    os.remove(filepath)

    return (score,cnn_model)






'''
Train a transformed CNN
'''
def execute(cnn_model, dataset, output_dim=10,
            lr=0.001, epochs=100, batch_size=32,
            verbose=False, patience=20, test=True, 
            log_path="none", baseline=False, upload=True, class_curve=False, 
            save=False, data_augmentation=True, tensorBoard=False):

    # self.gpu_limit()   
    log = False
    if(log_path != "none"):
        log = True

    past_dim = len(cnn_model.weights[-1][1])
    if(output_dim > past_dim):
        cnn_model = CNN_transformation.outLayer_transform(cnn_model, output_dim, avg=False)
    if type(dataset.train_images) is not np.ndarray:
        raise ValueError('The dataset is not in numpy ndarray format')
    architecture = cnn_model.architecture

    x_train, y_train = dataset.train_images, dataset.train_labels
    x_val, y_val = dataset.validation_images, dataset.validation_labels
    x_test, y_test = dataset.test_images, dataset.test_labels

    input_shape = x_train.shape[1:]

    '''
    Build the model 
    '''
    filepath=cnn_model.name + "execute.best.hdf5"
    model = build_model(cnn_model, architecture, input_shape, output_dim)


    '''
    finish the tasks for all newly added identity layer BN 
    '''
    if(len(cnn_model.tasks) > 0):
        raise ValueError('please Update all newly added BN before execution')

    '''
    Compile the Model
    '''
    model = build_model(cnn_model, architecture, input_shape, output_dim)
    model, CallList = compile_model(cnn_model, model, lr, filepath, verbose,
                patience, log_path, log=log, class_curve=class_curve, baseline=baseline, tensorBoard=tensorBoard)

    '''
    training process 
    '''
    model = train_model(model, x_train, y_train, x_val, y_val, epochs=epochs, 
            batch_size=batch_size, verbose=verbose, callbacks=CallList, 
            shuffle=True, data_augmentation=data_augmentation)

    if (epochs > 0):
        model.load_weights(filepath)    
    score = 0
    if(test):
        score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    else:
        score = model.evaluate(x_val, y_val, batch_size=batch_size,verbose=0)

    cnn_model.numParams = model.count_params()

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
        cnn_model.weights = weights
    if(save):
        #model.save(cnn_model.name + str(output_dim) + '.h5')
        save_object(cnn_model, cnn_model.name + str(output_dim) + '.pkl')
        print (cnn_model.name + str(output_dim) + '.pkl' + 'saved to disk')

    os.remove(filepath)       #remove the stored weight file 
    return (score,cnn_model)

'''
build a baseline that is retrained from scratch
'''
def baseline(cnn_model, dataset, output_dim=10,
            lr=0.001, epochs=100, batch_size=32,
            verbose=False, patience=20,test=True, log_path="none",
            upload=False,class_curve=False, save=False, data_augmentation=True):

    (score,cnn_model) = execute(cnn_model, dataset, output_dim=output_dim,
            lr=lr, epochs=epochs, batch_size=batch_size, verbose=verbose,
            patience=patience, test=test, log_path=log_path, baseline=True,
            upload=upload, class_curve=class_curve, save=save, data_augmentation=data_augmentation)
    return (score,cnn_model)

'''
returns the predictions made
'''

def class_evaluation(cnn_model, dataset,
    output_dim=10, lr=0.001, batch_size=32,
    verbose=False):

    # self.gpu_limit()
    past_dim = len(cnn_model.weights[-1][1])
    if(output_dim > past_dim):
        cnn_model = CNN_transformation.outLayer_transform(cnn_model, output_dim, avg=False)
    x_test,y_test = dataset.test_images, dataset.test_labels
    print (len(x_test))

    architecture = cnn_model.architecture
    input_shape = x_test.shape[1:]

    filepath = "not_really.path"
    model = build_model(cnn_model, architecture, input_shape, output_dim)
    model, CallList = compile_model(cnn_model, model, lr, filepath, verbose,
                20, "none", log=False, class_curve=False, baseline=False)

    predictions = model.predict(x_test,verbose=verbose)
    class_count = np.zeros((output_dim,2))
    for i in range(0,len(predictions)):
        correct = np.argmax(y_test[i])
        class_count[correct][1] = class_count[correct][1] + 1
        if( (np.argmax(predictions[i])) == correct):
            class_count[correct][0] = class_count[correct][0] + 1

    for i in range(0,output_dim):
        print ("class " + str(i+1) + " : " + str(class_count[i][0]) + " / " + str(class_count[i][1]))
        # accuracy = float(class_count[i][0]) / float(class_count[i][1])
        # print ("test accuracy is " + str(accuracy) )

    score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    print ("overall average test accuracy is " + str(score[1]))
    return (predictions,score)

def get_val_accu(cnn_model, dataset,
    output_dim=10, lr=0.001, batch_size=32,
    verbose=False):

    past_dim = len(cnn_model.weights[-1][1])
    if(output_dim > past_dim):
        cnn_model = CNN_transformation.outLayer_transform(cnn_model, output_dim, avg=False)
    x_val,y_val = dataset.validation_images, dataset.validation_labels
    architecture = cnn_model.architecture
    input_shape = x_val.shape[1:]

    filepath = "not_really.path"
    model = build_model(cnn_model, architecture, input_shape, output_dim)
    model, CallList = compile_model(cnn_model, model, lr, filepath, verbose,
                20, "none", log=False, class_curve=False, baseline=False)

    score = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
    return score[1]
    


'''
subsampling by random for each epoch
ratio indicates the how much previous data is used compared to the new dataset
'''
def random_sample_by_epoch(model, prev_dataset, new_dataset, 
                        ratio, filepath, log_path, epochs, patience, 
                        log=False, test=True, batch_size=32, method="randomReplace"):

    '''
    prepare datasets
    '''
    if (log):
        f = open(log_path+"accuracy.txt", "w")
        f.write("epoch, new val accuracy, old val accuracy, overall val accuracy"+'\n')

    print("epoch, new val accuracy, old val accuracy, overall val accuracy"+'\n')
    x_train_old = np.asarray(prev_dataset.train_images)
    y_train_old = np.asarray(prev_dataset.train_labels)
    x_train_new = np.asarray(new_dataset.train_images)
    y_train_new = np.asarray(new_dataset.train_labels)
    x_val_old = np.asarray(prev_dataset.validation_images)
    y_val_old = np.asarray(prev_dataset.validation_labels)
    x_val_new = np.asarray(new_dataset.validation_images)
    y_val_new = np.asarray(new_dataset.validation_labels)

    #check the accuracy of the current model before training
    new_accuracy = model.test_on_batch(x_val_new, y_val_new)
    old_accuracy = model.test_on_batch(x_val_old, y_val_old)
    print ("before curriculum training, model achieved validation accuracy "+ str(new_accuracy[1]) + "on new dataset")
    print ("before curriculum training, model achieved validation accuracy "+ str(old_accuracy[1]) + "on past dataset")

    lenEpoch = int(ratio * len(y_train_new))
    print ("each epoch contains " + str(lenEpoch) + " past data")
    print ("each epoch contains " + str(len(y_train_new)) + " new data")
    lenTotal = len(y_train_old)
    curriculum = np.zeros(len(y_train_old))
    executeNum = 0
    counter = 0
    best_accu = 0
    for i in range(0,epochs):
        '''
        training phase
        '''
        #obtain the mask for this epoch
        if (method == "randomReplace"):
            mask = Curriculum.random_with_replacement(lenEpoch, lenTotal)
        elif (method == "randomNoReplace"):
            (mask, curriculum, executeNum) = Curriculum.random_curriculum(lenEpoch, lenTotal, executeNum, curriculum)

        this_epoch_images = x_train_old[mask]
        this_epoch_labels = y_train_old[mask]
        this_epoch_images = np.concatenate((this_epoch_images,x_train_new),axis=0)
        # print(this_epoch_labels.shape)
        # print(y_train_new.shape)
        this_epoch_labels = np.concatenate((this_epoch_labels,y_train_new),axis=0)
        #shuffle old and new
        idx = np.random.permutation(len(this_epoch_images))
        this_epoch_images = this_epoch_images[idx]
        this_epoch_labels = this_epoch_labels[idx]
        '''
        further divide into mini-batch
        '''
        numLeft = len(this_epoch_images)
        index = 0
        while (numLeft > 0):
            if (numLeft < batch_size):
                model.train_on_batch(this_epoch_images[index:index + numLeft], this_epoch_labels[index:index + numLeft])
            else:
                model.train_on_batch(this_epoch_images[index:index + batch_size], this_epoch_labels[index:index + batch_size])
            numLeft = numLeft - batch_size

        '''
        test accuracies
        '''
        new_accuracy = model.test_on_batch(x_val_new, y_val_new)
        old_accuracy = model.test_on_batch(x_val_old, y_val_old)
        x_overall = np.concatenate((x_val_new, x_val_old),axis=0)
        y_overall = np.concatenate((y_val_new, y_val_old),axis=0)
        overall = model.test_on_batch(x_overall,y_overall)
        print (str(i)+","+str(new_accuracy[1])+','+str(old_accuracy[1])+','+str(overall[1]) + '\n')
        if(log):
            f.write(str(i)+","+str(new_accuracy[1])+','+str(old_accuracy[1])+','+str(overall[1]) + '\n')

        #implementing check pointing base on overall accuracy 
        if(overall[1] > best_accu):
            model.save_weights(filepath)
            best_accu = overall[1]
            counter = 0
        if(overall[1] < best_accu):
            counter = counter + 1
        if(counter > patience):
            break

    model.load_weights(filepath)
    score = 0
    if(test):
        x_test_old = np.asarray(prev_dataset.test_images)
        y_test_old = np.asarray(prev_dataset.test_labels)
        x_test_new = np.asarray(new_dataset.test_images)
        y_test_new = np.asarray(new_dataset.test_labels)
        x_overall = np.concatenate((x_test_new, x_test_old),axis=0)
        y_overall = np.concatenate((y_test_new, y_test_old),axis=0)
        score = model.evaluate(x_overall, y_overall, batch_size=batch_size, verbose=0)
    else:
        x_overall = np.concatenate((x_val_new, x_val_old),axis=0)
        y_overall = np.concatenate((y_val_new, y_val_old),axis=0)
        score = model.evaluate(x_overall, y_overall, batch_size=batch_size, verbose=0)
    return (score, model)

'''
curriculum_learning uses train_on_batch in keras and gives more freedom to what samples are fed each epoch
curriculum is a string that indicates the method used for designing the curriculum
currently only support: "random"
ratio indicates the how much previous data is used compared to the new dataset
recommend: 1.0
which means if there are x new samples,x old samples will be randomly selected
'''
def curriculum_learning(cnn_model, prev_dataset, new_dataset, 
    method="randomNoReplace", ratio=1.0, output_dim=10, 
    lr=0.001, epochs=100, batch_size=32, verbose=False, 
    patience=20,test=True, log_path="none", 
    baseline=False, upload=True, class_curve=False, save=False):
    
    '''
    set up parameters and dataset
    '''
    log = False
    if(log_path != "none"):
        log = True
    
    past_dim = len(cnn_model.weights[-1][1])
    if(output_dim > past_dim):
        cnn_model = CNN_transformation.outLayer_transform(cnn_model, output_dim,avg=False)
        (prev_dataset,new_dataset) = datasets.transform_dataset(prev_dataset, new_dataset)
    architecture = cnn_model.architecture
    input_shape = prev_dataset.train_images.shape[1:]

    '''
    Compile the model 
    '''
    filepath=cnn_model.name + "execute.best.hdf5"
    model = build_model(cnn_model, architecture, input_shape, output_dim)
    model, CallList = compile_model(cnn_model, model, lr, filepath, verbose,
                patience, log_path, log=log, class_curve=class_curve, baseline=baseline)
    '''
    train by epoch
    '''
    score = 0
    (score, model) = random_sample_by_epoch(model, prev_dataset, new_dataset, 
                        ratio, filepath, log_path, epochs, patience, 
                        log=log, test=test, batch_size=batch_size, method=method)

    if(save):
        #model.save(cnn_model.name + str(output_dim) + '.h5')
        save_object(cnn_model, cnn_model.name + str(output_dim) + '.pkl')
        print (cnn_model.name + str(output_dim) + '.pkl' + 'saved to disk')
    return (score,cnn_model)


def layer_outputs(cnn_model, dataset, layerList, CompiledModel,
                    output_dim, batch_size, data_augmentation):
    
    architecture = cnn_model.architecture

    model = CompiledModel
    TaskList = layerList
    x_train, y_train = dataset.train_images, dataset.train_labels
    x_val, y_val = dataset.validation_images, dataset.validation_labels
    x_test, y_test = dataset.test_images, dataset.test_labels
    #concatenate validation data with training data if needed
    input_shape = x_test.shape[1:]

    inp = model.input                                           # input placeholder
    taskLayers = [model.get_layer(index=task) for task in TaskList] #select the layers that we need to calculate statistics
    outputs = [layer.output for layer in taskLayers]          # tasked layer outputs
    functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions


    x_batches = []
    y_batches = []  #don't really need y_batch but still put here to use with generator

    for cbatch in range(0, x_test.shape[0], batch_size):
        x_batches.append(x_train[cbatch:(cbatch + batch_size),:,:])

    batch_num = len(x_batches)
    statistics = [[0, 0] for _ in TaskList]
        
    layer_outs = [func([x_batches[0]]) for func in functors] #one batch at a time
    layer_outs = np.asarray(layer_outs)

    taskLayers = [model.get_layer(index=task+1) for task in TaskList] #select the layers that we need to calculate statistics

    print("examine the following layers:")
    print(TaskList)

    count = 0
    for out in layer_outs:
        print("layer " + str(TaskList[count]))
        count = count + 1
        print(out.shape)
        print(out[0][0][1][1])

    #find empty lists within architecture
    archIdx = []
    for _i in range(0, len(architecture)):
        #unfinished task
        if (len(architecture[_i].size) == 0):
            archIdx.append(_i)

    cnn_model.tasks = []        #task completed
    return cnn_model





















def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    return pickle.load( open( filename, "rb" ) )



















































