#currently contains dataset processing for 
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras import utils
import random
import math
import numpy as np



'''
Dataset contains training set/validation set/test set
Usually one Dataset object only contain all data from one class
Only modification to DataSet is through concatenate method
'''

class DataSet(object):
    def __init__(self,train_images,train_labels,
                validation_images,validation_labels,
                test_images,test_labels):
        self._train_images = train_images
        self._train_labels = train_labels
        self._validation_images = validation_images
        self._validation_labels = validation_labels
        self._test_images = test_images
        self._test_labels = test_labels
        self._tr_index = 0
        self._val_index = 0
        self._tr_length = len(self._train_images)
        self._val_length = len(self._validation_images)

    @property
    def train_images(self):
        return self._train_images

    @property
    def train_labels(self):
        return self._train_labels

    @property
    def validation_images(self):
        return self._validation_images

    @property
    def validation_labels(self):
        return self._validation_labels

    @property
    def test_images(self):
        return self._test_images

    @property
    def test_labels(self):
        return self._test_labels

    def set_train_images(self,train_images):
        self._train_images = train_images

    def set_train_labels(self,train_labels):
        self._train_labels = train_labels

    def set_validation_images(self,validation_images):
        self._validation_images = validation_images

    def set_validation_labels(self,validation_labels):
        self._validation_labels = validation_labels

    def set_test_images(self,test_images):
        self._test_images = test_images

    def set_test_labels(self,test_labels):
        self._test_labels = test_labels

    
    '''
    concatenate training,validation or test set
    concatenate with existing class
    percentage is a float: if not specified, its default value is 1.0
    '''
    def concatenate(self,indicator,new_images,new_labels):
        if(indicator == "train"):
            self._train_images = np.concatenate((self._train_images,new_images),axis=0)
            self._train_labels = np.concatenate((self._train_labels,new_labels),axis=0)
            self.shuffle(indicator)
        elif(indicator == "validation"):
            self._validation_images = np.concatenate((self._validation_images,new_images),axis=0)
            self._validation_labels = np.concatenate((self._validation_labels,new_labels),axis=0)
            self.shuffle(indicator)
        elif(indicator == "test"):      
            self._test_images = np.concatenate((self._test_images,new_images),axis=0)
            self._test_labels = np.concatenate((self._test_labels,new_labels),axis=0)
        else:
            raise ValueError('can only concatenate train, val or test set')

    '''
    randomly shuffle a set so it is not in its default order
    '''
    def shuffle(self,indicator):
        if(indicator == "train"):
            order = random.sample(range(0,len(self._train_images)),len(self._train_images))
            shuffle_images=[]
            shuffle_labels=[]
            for i in range(0,len(self._train_images)):
                shuffle_images.append(self._train_images[order[i]])
                shuffle_labels.append(self._train_labels[order[i]])
            self._train_images = np.asarray(shuffle_images)
            self._train_labels = np.asarray(shuffle_labels)
        elif(indicator == "validation"):
            order = random.sample(range(0,len(self._validation_images)),len(self._validation_images))
            shuffle_images=[]
            shuffle_labels=[]
            for i in range(0,len(self._validation_images)):
                shuffle_images.append(self._validation_images[order[i]])
                shuffle_labels.append(self._validation_labels[order[i]])

            self._validation_images = np.asarray(shuffle_images)
            self._validation_labels = np.asarray(shuffle_labels)
        elif(indicator == "test"):
            order = random.sample(range(0,len(self._test_images)),len(self._test_images))
            shuffle_images=[]
            shuffle_labels=[]
            for i in range(0,len(self._test_images)):
                shuffle_images.append(self._test_images[order[i]])
                shuffle_labels.append(self._test_labels[order[i]])

            self._test_images = np.asarray(shuffle_images)
            self._test_labels = np.asarray(shuffle_labels)
        else:
            raise ValueError('can only shuffle train, val or test set')

    '''
    Split an existing class into partial classes and keeps track of the progress
    1. training examples are split based on the partition (so does validation examples)
    2. testing examples are all given at once for compariable test results
    3. if requested propotion > remaining, report error 
    '''
    def split(self, percentage):
        tr_partial = int(self._tr_length*percentage-1)
        if (self._tr_length-1-self._tr_index) < 0:
            print("requested examples: " + str(tr_partial))
            print("remaining examples: " + str(self._tr_length-1-self._tr_index))
            raise ValueError('requested more than remaining')

        #check if test data is already given out
        tested = False
        if (self._tr_index > 0):
            tested = True

        #construct a new dataset object to return
        requested_tr_images = self._train_images[self._tr_index:self._tr_index + tr_partial]
        requested_tr_labels = self._train_labels[self._tr_index:self._tr_index + tr_partial]
        self._tr_index = self._tr_index + tr_partial
        val_partial = int(self._val_length*percentage-1)
        requested_val_images = self._validation_images[self._val_index:self._val_index + val_partial]
        requested_val_labels = self._validation_labels[self._val_index:self._val_index + val_partial]
        self._val_index = self._val_index + val_partial
        if (not tested):
            requested_test_images = self._test_images
            requested_test_labels = self._test_labels
        else:
            requested_test_images = []
            requested_test_labels = []

        return (DataSet(requested_tr_images,requested_tr_labels,requested_val_images,requested_val_labels,requested_test_images,requested_test_labels), tested)




'''
a simple function to create empty 2d lists
'''
def create_2d(num_classes):
    temp=[]
    for i in range(0,num_classes):
        temp.append([])
    return temp


'''
swap label number between 2 classes 
'''
def label_swap(y,labelA,labelB):
    for i in range(len(y)):
        if (y[i] == labelA):
            y[i] = labelB
        elif (y[i] == labelB):
            y[i] = labelA
    return y



'''
partition a dataset into individual classes stored in Dataset objects
data_str: a string to indicate which dataset to partition
cifar10, cifar100, mnist, fashion_mnist
val_split: indicates how much data is used as validation set
order: the ordering of the classes
normalization: which dataset normalization method to use
labelSwap is a list of 2-element lists each contains the class labels to swap with 

return Datasets list contains each class in a Dataset object
'''
def Incremental_partition(data_str, val_split, order, normalization="numerical", labelSwap=None):
    num_classes = 10

    '''
    load raw data from keras
    '''
    if(data_str == "mnist"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif(data_str == "fashion_mnist"):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        num_classes = 10
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    elif(data_str == "cifar10"):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
    elif(data_str == "cifar100"):
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        num_classes = 100
    else:
        raise ValueError('the requested dataset is currently not available')

    x_train,x_test = normalize(x_train, x_test, normalization)

    if(labelSwap != None):
        for pair in labelSwap:
            y_train = label_swap(y_train, pair[0], pair[1])
            y_test = label_swap(y_test, pair[0], pair[1])

    '''
    flatten training set / test set for later reordering
    '''
    training_images = []
    training_labels = []
    for i in range(0,num_classes):
        idx = np.where(y_train == i)[0]
        class_images = np.asarray([x_train[j] for j in idx])
        training_images.append(class_images)
        class_labels = np.asarray([y_train[j] for j in idx])
        class_labels = class_labels.flatten()
        training_labels.append(class_labels)

    testing_images = []
    testing_labels = []
    for i in range(0,num_classes):
        idx = np.where(y_test == i)[0]
        class_images = np.asarray([x_test[j] for j in idx])
        testing_images.append(class_images)
        class_labels = np.asarray([y_test[j] for j in idx])
        class_labels = class_labels.flatten()
        testing_labels.append(class_labels)

    '''
    switch class order for both training set and test set
    '''
    switch_images = create_2d(num_classes)
    switch_labels = create_2d(num_classes)
    for i in range(0,len(order)):
        switch_images[i] = training_images[order[i]]
        switch_labels[i] = [i]*len(training_labels[order[i]])
    training_images = switch_images
    training_labels = switch_labels

    switch_images = create_2d(num_classes)
    switch_labels = create_2d(num_classes)
    for i in range(0,len(order)):
        switch_images[i] = testing_images[order[i]]
        switch_labels[i] = [i]*len(testing_labels[order[i]])
    testing_images = switch_images
    testing_labels = switch_labels

    return split_classes(num_classes,training_images,training_labels,testing_images,testing_labels,val_split)


'''
MNIST classes are 0-9
Fashion MNIST classes are 10-19
'''

def All_MNISTlike(val_split, normalization="numerical", mix=False):
    (x1_train, y1_train), (x1_test, y1_test) = mnist.load_data()
    (x2_train, y2_train), (x2_test, y2_test) = fashion_mnist.load_data()

    num_classes = 20

    x1_train = x1_train.reshape(60000, 784)
    x1_test = x1_test.reshape(10000, 784)
    x2_train = x2_train.reshape(60000, 784)
    x2_test = x2_test.reshape(10000, 784)

    #let fashion mnist be classes 10-19
    if (not mix):
        y2_train = y2_train + 10 
        y2_test = y2_test + 10
    else:
        y1_train = y1_train * 2         #even number classes are MNIST
        y1_test = y1_test * 2
        y2_train = y2_train * 2 + 1     #odd number classes are Fashion-MNIST
        y2_test = y2_test * 2 + 1           

    x1_train,x1_test = normalize(x1_train, x1_test, normalization)
    x2_train,x2_test = normalize(x2_train, x2_test, normalization)


    training_images = create_2d(num_classes)
    training_labels = create_2d(num_classes)
    for i in range(0,len(y1_train)):
        label = y1_train[i]
        training_images[label].append(x1_train[i])
        training_labels[label].append(y1_train[i])

    for i in range(0,len(y2_train)):
        label = y2_train[i]
        training_images[label].append(x2_train[i])
        training_labels[label].append(y2_train[i])


    testing_images = create_2d(num_classes)
    testing_labels = create_2d(num_classes)
    for i in range(0,len(y1_test)):
        label = y1_test[i]
        testing_images[label].append(x1_test[i])
        testing_labels[label].append(y1_test[i])
    for i in range(0,len(y2_test)):
        label = y2_test[i]
        testing_images[label].append(x2_test[i])
        testing_labels[label].append(y2_test[i])
    
    return split_classes(num_classes,training_images,training_labels,testing_images,testing_labels,val_split)

'''
create empty 2d list
'''
def create_2d(num_classes):
    temp=[]
    for i in range(0,num_classes):
        temp.append([])
    return temp


'''
To normalize or preprocess the dataset before partitioning
'''
def normalize(x_train, x_test, technique):
    '''
    doing normalization first
    '''
    if(technique == "numerical"):
        x_train = (x_train / 128 -1)
        x_test = (x_test / 128 -1)
    elif(technique == "std"):
        S = np.std(x_train,axis=0)
        M = np.mean(x_train,axis=0)
        for i in range(0,len(S)):
            if (S[i] == 0):
                S[i] = 1.0
        x_train = (x_train - M) / S
        x_test = (x_test - M) / S
    else:
        x_train = x_train / 255
        x_test = x_test / 255

    return (x_train,x_test)


'''
split training set into training set and validation set
return a list of Dataset Object
'''
def split_classes(num_classes,training_images,training_labels,testing_images,testing_labels,val_split):

    datasets = []
    for i in range(0,num_classes):
        num_labels = i+1
        val_num = int(val_split * len(training_images[i]))
        if(i==0):
            num_labels = 2

        onehot_train = utils.to_categorical(training_labels[i],num_labels)
        onehot_test = utils.to_categorical(testing_labels[i],num_labels)
        x_train = training_images[i][val_num:-1]
        y_train = onehot_train[val_num:-1]
        x_val = training_images[i][0:val_num]
        y_val = onehot_train[0:val_num]
        x_test = testing_images[i]
        y_test = onehot_test

        datasets.append(DataSet(x_train,y_train,x_val,y_val,x_test,y_test))

    return datasets


'''
Sum two Dataset objects
the summed Dataset object is stored in the old index
percentage indicate how much past data is concatenated
default is 100%
'''
def sum_data(datasets,new_index,old_index,percentage=1.0):
    old_label_num = len(datasets[old_index].train_labels[0])
    new_label_num = len(datasets[new_index].train_labels[0])
    if(old_label_num!=new_label_num):
        (datasets[old_index], datasets[new_index]) = transform_dataset(datasets[old_index], datasets[new_index])

    if(percentage != 1.0):
        (newDataset,tested) = datasets[new_index].split(percentage)
    else:
        newDataset = datasets[new_index]

    datasets[old_index].concatenate("train", newDataset.train_images, newDataset.train_labels)
    datasets[old_index].concatenate("validation", newDataset.validation_images, newDataset.validation_labels)
    datasets[old_index].concatenate("test", newDataset.test_images, newDataset.test_labels)
    return datasets

'''     
transforms one hot encoding of dataset at old_index
'''
def transform_dataset(old_dataset, new_dataset):
    past_num = len(old_dataset.train_labels[0])
    new_num = len(new_dataset.train_labels[0])

    if(past_num > new_num):     #new dataset has less dimension in onehot encoding labels
        fzero = np.zeros((len(new_dataset.train_labels),past_num-new_num),dtype=(new_dataset.train_labels[0].dtype))
        train_labels = np.concatenate((new_dataset.train_labels,fzero),axis=1)
        train_images = new_dataset.train_images
        new_dataset.set_train_labels(train_labels)

        fzero = np.zeros((len(new_dataset.validation_labels),past_num-new_num),dtype=(new_dataset.validation_labels[0].dtype))
        validation_labels = np.concatenate((new_dataset.validation_labels,fzero),axis=1)
        new_dataset.set_validation_labels(validation_labels)

        fzero = np.zeros((len(new_dataset.test_labels),past_num-new_num),dtype=(new_dataset.test_labels[0].dtype))
        test_labels = np.concatenate((new_dataset.test_labels,fzero),axis=1)
        new_dataset.set_test_labels(test_labels)
        

    if(past_num < new_num):
        fzero = np.zeros((len(old_dataset.train_labels),new_num-past_num),dtype=(old_dataset.train_labels[0].dtype))
        train_labels = np.concatenate((old_dataset.train_labels,fzero),axis=1)
        train_images = old_dataset.train_images
        old_dataset.set_train_labels(train_labels)

        fzero = np.zeros((len(old_dataset.validation_labels),new_num-past_num),dtype=(old_dataset.validation_labels[0].dtype))
        validation_labels = np.concatenate((old_dataset.validation_labels,fzero),axis=1)
        old_dataset.set_validation_labels(validation_labels)

        fzero = np.zeros((len(old_dataset.test_labels),new_num-past_num),dtype=(old_dataset.test_labels[0].dtype))
        test_labels = np.concatenate((old_dataset.test_labels,fzero),axis=1)
        old_dataset.set_test_labels(test_labels)

    return (old_dataset, new_dataset)




'''
generate a variable k class arrival order (Integer)
minRange <= k <= maxRange
output: a list contains the number of classes for each step
'''
def variable_K_integer(total, minRange, maxRange, initial_classes=None):
    order = []
    Remaining = total
    index = 0

    if (total < maxRange):
        raise ValueError('cant have maxRange bigger than total')

    if (initial_classes is not None):
        for element in initial_classes:
            Remaining = Remaining - 1
            index = index + 1
        order.append(initial_classes[:])
    else:
        pass


    while (Remaining > maxRange):
        outcome = random.randint(minRange,maxRange)
        Remaining = Remaining - outcome
        classes = []
        for i in range(0,outcome):
            classes.append(index)
            index = index + 1 
        order.append(classes[:])

    classes = []
    for i in range(0,Remaining):
        classes.append(index)
        index = index + 1 
    order.append(classes[:])

    return order



def MLP_ordering(total, minRange, maxRange, initial_classes=None):
    order = []
    Remaining = total
    ClassCount = [1.0]*total

    if (total < maxRange):
        raise ValueError('cant have maxRange bigger than total')

    if (initial_classes is not None):
        for element in initial_classes:
            Remaining = Remaining - 1
            ClassCount[element] = 0.0
        order.append(initial_classes[:])
    else:
        pass


    while (Remaining > maxRange):
        outcome = random.randint(minRange,maxRange)
        Remaining = Remaining - outcome
        classes = []
        for i in range(0,outcome):
            classes.append(find_integer(ClassCount))
        order.append(classes[:])

    classes = []
    for i in range(0,Remaining):
        classes.append(find_integer(ClassCount))
    order.append(classes[:])
    return order





'''
generate a variable k class arrival order (Fraction)
initial_classes arguments inidcates the initial classes thus will be set to 0.0 in fractionList
1 class can be divided into multiple subsets and arrive at different time steps
the fractions that are considered here are only from fractionList
each step will randomly select an integer within Range and a fraction from fractionList
minRange <= K <= maxRange
output: a list of tuples that contains instructions for class arrival at each step
each tuple (x,y,z) indicates: x number of new classes (Integer), y amount of partial class arrived (fraction), z the class number of that partial class
z -1 means there are no partial class for this step 
'''
def variable_K_fraction(total, fractionList, minRange, maxRange, initial_classes=None):
    order = []
    Remaining = total
    fractionCount = [1.0]*total

    if (initial_classes is not None):
        for element in initial_classes:
            fractionCount[element] = 0.0
        order.append((initial_classes[:],0.0,-1))
    else:
        pass

    if (total < maxRange):
        raise ValueError('cant have maxRange bigger than total')

    while (Remaining > maxRange):
        # select integer output 
        x = random.randint(int(minRange),maxRange)
        Remaining = Remaining - x
        classes = []
        for i in range(0,x):
            classes.append(find_integer(fractionCount))
        y = float(random.choice(fractionList))
        z = find_fraction(fractionCount)
        if (z >= 0):
            if (fractionCount[z] <= (0.2)):
                y = fractionCount[z]    #clean up the fractions
            fractionCount[z] = fractionCount[z] - y
            Remaining = Remaining - y
            order.append((classes[:],y,z))
        else:   #no more partial classes
            Remaining = 0.0
            y = 0.0
            if len(classes) > 0:
                order.append((classes[:],y,z))
            break

    #when Remaining <= maxRange
    x=int(Remaining)
    classes = []
    for i in range(0,x):
        classes.append(find_integer(fractionCount))
    check = []
    for value in classes:
        if value is not None:
            check.append(value)
    classes = check
    y=Remaining - x
    z=find_fraction(fractionCount)
    if z >= 0 and len(classes) > 0:
        order.append((classes[:],y,z))
    return order


'''
find the index of the first non-zero element  
'''
def find_fraction(fractionCount):
    for i in range(0, len(fractionCount)):
        if fractionCount[i] > 0.0:
            return i
    return (-1)

'''
find index of the first 1.0
'''
def find_integer(fractionCount):
    for i in range(0, len(fractionCount)):
        if (fractionCount[i] == 1.0):
            fractionCount[i] = 0.0      #added this class
            return i















