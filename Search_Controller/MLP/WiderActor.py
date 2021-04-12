from keras.models import Model
from keras.models import load_model
from keras.layers import LSTM, Bidirectional, Embedding, concatenate, RepeatVector, TimeDistributed, Dense, Input, Lambda
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
import tensorflow as tf
import numpy as np



class WiderActor:
    
    def __init__(self, num_input, max_actions, entropy_penalty=0.1, lr=0.001):
        self.num_input = num_input
        self.max_actions = max_actions
        self.entropy_penalty = entropy_penalty
        self.lr = lr
        self.model = None
        self.train_fn = None
        self.__build_network()
        self.__build_train_fn()
        self.name = "widerActor"



    def __build_network(self):
        '''
        [Validation difference, number of conv layers, number of fc layers]
        '''
        inputs = Input(shape=(self.num_input,))         #input shape: [batch_size, num_input]
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        probs = Dense(self.max_actions, activation='softmax')(x)    #softmax provides probability for each action 
        print(probs.shape)
        self.model = Model(inputs=inputs, outputs=probs)
        return self.model

   

    def __build_train_fn(self):
        '''
        replaces model.fit() function with train_fn
        '''
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.max_actions))         #[batch_size, max_actions]
        discount_reward_placeholder = K.placeholder(shape=(None,))     #[batch_size, 1]

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        entropy = -K.sum(action_prob_placeholder * K.log(action_prob_placeholder))      #add an entropy regularizer to encourage exploration
        loss = loss - self.entropy_penalty * entropy
        
        adam = optimizers.Adam(lr=self.lr)

        updates = adam.get_updates(params=self.model.trainable_weights, loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)

    '''
    predict returns the probability produced by the model
    '''
    def predict(self, state_batch):
        return self.model.predict(state_batch)

    '''
    generate an action by provided probability from the network 
    input : state_batch, (n_samples, state_dimension)
    output: actions, (n_samples,1)
    '''
    def get_action(self, state_batch):
        action_probs = self.model.predict(state_batch)
        actions = []

        for action_prob in action_probs:
            actions.append(np.random.choice(np.arange(self.max_actions), p=action_prob))
        return actions


    """
    Update the policy 
    Args:
        state_batch (2-D Array): `state` array of shape (n_samples, state_dimension)
        actions (1-D Array): `action` array of shape (n_samples,)
            It's simply a list of int that stores which actions the agent chose
        rewards (1-D Array): `reward` array of shape (n_samples,)
            A reward is given after each action.
    """
    def fit(self, state_batch, actions, rewards):
        action_onehot = np_utils.to_categorical(actions, num_classes=self.max_actions)
        self.train_fn([state_batch, action_onehot, rewards])

    def save_model(self):
        self.model.save(self.name)

    def load_model(self):
        self.model = load_model(self.name)
        self.__build_train_fn()

    '''
    input shape: [batch_size, max_actions]
    output shape: [batch_size, max_actions]
    '''
    def tf_log(self, x):
        x = tf.log(x) 
        return tf.cast(x, tf.float32)


    '''
    input shape: [batch_size, max_actions]
    output shape: [batch_size, max_actions,1]
    '''
    def tf_multinomial(self, logits):
        decision = tf.random.categorical(logits, 1)
        return tf.cast(decision, tf.float32)







