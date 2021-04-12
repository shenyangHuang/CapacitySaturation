from Architecture_Embedding import CNNarch_Embedding
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Embedding, concatenate, RepeatVector, TimeDistributed, Dense, Input, Lambda
import tensorflow as tf
import numpy as np

'''
For a single batch, you must have the sample number of timesteps (which can be achieved through 0 padding). Because input needs to be a tensor
For different batches, you can have different number of timesteps 
https://stackoverflow.com/questions/45649520/explain-with-example-how-embedding-layers-in-keras-works
Input of the EncoderNet should be a list of layers of the current CNN architecture
thus a list of vocab of layers 

num_step indicates the maximum possible number of layers in the CNN
embedding_dim indicates the dimension of the embedded low-d vector
'''
#recommended num_steps is 50 
class EncoderNet:
    def __init__(self, num_steps, embedding_dim, size_per_layer=50, vocab="cnn"):
        self.num_steps = num_steps
        self.embedding_dim = embedding_dim
        self.size_per_layer = size_per_layer
        self.vocab = None
        self.output = None
        self.input = None
        if (vocab == "cnn"):
            self.vocab = CNNarch_Embedding.define_cnn_vocab()

    def build(self):
        #vocabulary size defines how many possible CNN layer configurations there are 
        #layer embedding


        '''
        Use Keras functional API
        '''

        layer_tokens = Input(shape=(self.num_steps,), name='layer_tokens')         #input shape: [batch_size, num_steps]
        embedded_arch = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab.size, input_length=self.num_steps)(layer_tokens)
        print (embedded_arch.shape)
        #output shape: [batch_size, num_steps, embedding_dim]
        '''
        NEED TO WRAP ALL TF FUNCTIONS INTO KERAS LAYERS
        '''

        embedded_arch = Lambda(self.tf_transpose)(embedded_arch)     #[num_steps, batch_size, embedding_dim]
        embedded_arch = Lambda(self.tf_reshape)((embedded_arch)) #[num_steps*batch_size, embedding_dim]
        embedded_arch = Lambda(self.tf_split)((embedded_arch))
        # embedded_arch = tf.transpose(embedded_arch, [1, 0, 2])           
        # embedded_arch = tf.reshape(embedded_arch, [-1, self.embedding_dim])         #[num_steps*batch_size, embedding_dim]
        # print (embedded_arch.shape)
        # embedded_arch = tf.split(embedded_arch, self.num_steps, 0)
        # print (len(embedded_arch))

        '''
        Bidirectional layer need input dimension of 3 which means its previous layer must have return_sequences=True
        RepeatVector layer need input dimension of 2 which means its previous layer must have return_sequences=False
        Encoder Input shape : [batch_size, num_steps, embedding_dim]
        Required shape: 'num_steps' tensors list of shape [batch_size, input_dim]
        '''
        #concatenate bidirectional layer output

        hidden_output = Bidirectional(LSTM(self.size_per_layer, activation='relu', return_sequences=True),merge_mode='concat', weights='None')(embedded_arch)
        
        print ("return sequences true")
        print (hidden_output.shape)

        # print ("hidden output shape")
        # print (len(hidden_output))
        # # (bi_outputs, forward_h, forward_c, backward_h, backward_c) = 
        # bi_outputs = hidden_output[0]
        # forward_h = hidden_output[1]
        # forward_c = hidden_output[2]
        # backward_h = hidden_output[3]
        # backward_c = hidden_output[4]
        # print ("bi_outputs shape")
        # print (bi_outputs.shape)
        # print ("forward_h shape")
        # print (forward_h.shape)
        # print ("forward_c shape")
        # print (forward_c.shape)



        # encoder_state_c = Lambda(tf_concat)((forward_c, backward_c))
        # encoder_state_h = Lambda(tf_concat)((forward_h, backward_h))
        # state = Lambda(tf_LSTMStateTuple)((encoder_state_c, encoder_state_h))

        # state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
        # bi_outputs = Lambda(tf_stack)(bi_outputs)

        #bi_outputs = tf.stack(bi_outputs, axis=0) #[num_steps, batch_size, rnn_units]
        # bi_outputs = Lambda(tf_transpose)(bi_outputs)

        #bi_outputs = tf.transpose(bi_outputs, [1, 0, 2]) # [batch_size, num_steps, rnn_units]
        # self.outputs = bi_outputs
        # self.states = state
        # self.inputs = layer_tokens
        # model = Model(inputs=layer_tokens, outputs=hidden_output)
        # self.model = model
        '''
        partially build a model but didn't compile
        '''
        self.output = hidden_output
        self.input = layer_tokens
        return ((self.output, self.input))

        # for i in range(self.num_layers-1):
        #     model.add(Bidirectional(LSTM(self.size_per_layer, activation='relu', return_sequences=True)))
        # model.add(Bidirectional(LSTM(self.size_per_layer, activation='relu')))
        # #get the output of the last bi-LSTM layer
        # model.add(RepeatVector(self.embedding_dim))
        # #add a decoder just for fun
        # model.add(Bidirectional(LSTM(self.size_per_layer, activation='relu', return_sequences=True)))
        # model.add(Bidirectional(LSTM(self.size_per_layer, activation='relu', return_sequences=True)))
        # model.add(TimeDistributed(Dense(self.vocab.size, activation='softmax')))
        # model.compile(loss='categorical_crossentropy', optimizer='adam')

        #output hidden states here
        # lstm, forward_h, forward_c, backward_h, backward_c = model.layers[-1].output
        # state_h = concatenate([forward_h, backward_h], axis=1)
        # state_c = concatenate([forward_c, backward_c], axis=1)
        # state = concatenate([state_h, state_c], axis=1)


    '''
    input shape: [batch_size, num_steps, embedding_dim]
    output shape: [num_steps, batch_size, embedding_dim]
    '''
    def tf_transpose(self, x):
        x = tf.transpose(x, [1, 0, 2]) 
        return tf.cast(x, tf.float32)


    '''
    input shape: [num_steps, batch_size, embedding_dim]
    output shape: [num_steps*batch_size, embedding_dim]
    '''
    def tf_reshape(self, embedded_arch):
        #embedding_dim = 16
        embedded_arch = tf.reshape(embedded_arch, [-1, self.embedding_dim]) 
        return tf.cast(embedded_arch, tf.float32)


    '''
    input shape: [num_steps*batch_size, embedding_dim]
    output shape: 'num_steps' tensors list of shape [batch_size, input_dim]
    '''
    def tf_split(self, embedded_arch):
        #num_steps = 50
        embedded_arch = tf.split(embedded_arch, self.num_steps, 0)
        return tf.cast(embedded_arch, tf.float32)


def tf_concat(forward, backward):
    encoder_state = tf.concat((forward, backward), axis=1)
    return tf.cast(encoder_state, tf.float32)

def tf_LSTMStateTuple(encoder_state_c, encoder_state_h):
    state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
    return tf.cast(state, tf.float32)

def tf_stack(x):
    x = tf.stack(x, axis=0)
    return tf.cast(x, tf.float32)












