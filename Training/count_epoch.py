import numpy as np
import keras
 
class count_epoch(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.num_epoch = 0
        return
 
    def on_train_end(self, logs={}):
        print("training used " + str(self.num_epoch) + " epochs")
        return
 
    def on_epoch_begin(self,epoch, logs={}):
        return 

    def on_epoch_end(self, epoch, logs={}):
        self.num_epoch = self.num_epoch + 1
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return