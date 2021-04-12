import random
import numpy as np
'''
maintains the memory for continual learning 
implements various sampling techniques to decide what stays in memory
memory is simply maintained as a list of (sample, label) pairs
'''

class Sampling(object):
    def __init__(self,size):
        self.size = size
        self.memory = []
        #n is the total number of samples observed so far
        self.n = 0
        self.numClass = 0


'''
1) Create an array reservoir[0..k-1] and copy first k items of stream[] to it.
2) Now one by one consider all items from (k+1)th item to nth item.
…a) Generate a random number from 0 to i where i is index of current item in stream[]. Let the generated random number is j.
…b) If j is in range 0 to k-1, replace reservoir[j] with arr[i]
implement similar to the tiny episodic memory paper
'''

class ResevoirSampling(Sampling):
    def update(self, dataset):
        x_train, y_train = dataset.train_images, dataset.train_labels
        self.numClass = y_train[0].shape[0]
        y_train = self.toInt(y_train)
        #shuffle the inputs first
        samples = self.shuffle(x_train, y_train)
        for j in range(0, len(samples)):
            '''
            check if memory is full
            '''
            if(len(self.memory) < self.size):
                self.memory.append(samples[j])
            else:
                '''
                update the resevoir
                '''
                i = random.randint(0, self.n + j)
                if (i < self.size):
                    self.memory[i] = samples[j]
        #update the number of samples observed so far
        self.n = self.n + len(samples)
        self.checkClasses()


    def shuffle(self, x_train, y_train):
        '''
        shuffle the new samples before sampling
        '''
        samples = []
        for i in range(0, len(x_train)):
            samples.append((x_train[i], y_train[i]))
        random.shuffle(samples)
        return samples

    '''
    convert training onehot to label
    '''
    def toInt(self, y_train):
        y_train = [np.where(r==1)[0][0] for r in y_train]
        return y_train



    #samples a minibatch for training
    #also transform to the correct output dimension with onehot embedding
    def minibatch(self, batch_size, outdim):
        random.shuffle(self.memory)
        ms = self.memory[0:batch_size]
        output = []
        for i in range(len(ms)):
            x = ms[i][0]
            y = np.zeros(outdim)
            y[ms[i][1]] = 1
            output.append((x,y))
        return output


    '''
    check how many samples per class you have in the memory
    '''
    def checkClasses(self):
        num_perClass = [0]*self.numClass
        for (x,y) in self.memory:
            num_perClass[y] = num_perClass[y] + 1

        print("check class balance in memory: ")
        for i in range(len(num_perClass)):
            print (" class " + str(i) + " has " + str(num_perClass[i]) + " samples")











