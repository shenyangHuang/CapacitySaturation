from Model import MLP_Model
from Model.Layer_obj import Layer_obj
from Search_Controller.MLP.WiderActor import WiderActor
from Search_Controller.MLP.deeperActor import deeperActor
from Search_Controller.RL.input_util import thermometer_encoding, decimal_encoding
import random
import numpy as np
import copy
import os


'''
This class is in charge of maintain WiderActor and DeeperActor which includes sampling and training
At each step generate a list of transformation instructions 

'''
class RL_director:

    '''
    automatically initializes widerActor and deeperActor
    '''
    def __init__(self, num_input, wider_max_actions, deeper_max_actions, max_layers, entropy_penalty=0.01, lr=0.001):
        self.num_input = num_input
        self.wider_max_actions = wider_max_actions
        self.deeper_max_actions = deeper_max_actions
        self.max_layers = max_layers
        self.entropy_penalty = entropy_penalty
        self.lr = lr
        self.widerActor = WiderActor(self.num_input, self.wider_max_actions, entropy_penalty=self.entropy_penalty, lr=self.lr)
        self.deeperActor = deeperActor(self.num_input, self.deeper_max_actions, entropy_penalty=self.entropy_penalty, lr=self.lr)
        self.prev_accu = 0
        self.widerNums = None
        self.deeperNums = None
        self.states = None


    '''
    generate a list of transformation instructions from WiderActor and DeeperActor
    requires the validation of current step after training with existing architecture
    max_duplicates indicates after how many duplicates are sampled, the algorithm will stop caring about duplicates (this is possible in early steps) 
    '''
    def RL_sample_step(self, architecture, sample_size, validation_accu, numNew, max_duplicate):
        val_diff = validation_accu - self.prev_accu
        if (self.prev_accu == 0):   #first training step set val_diff to be 0 as there is no previous accuracy to compare to 
            val_diff = 0
        states = self.state_gen(val_diff, architecture, sample_size, numNew)

        #obtain decisions from actor networks 
        widerNums = self.widerActor.get_action(states)
        deeperNums = self.deeperActor.get_action(states)

        #generate architectures based on decisions
        instructions = []
        counter = 0 
        for i in range(sample_size):
            equal = True 
            sample = None
            while(equal):
                equal = False
                sample = self._random_sample(architecture, widerNums[i], deeperNums[i])
                for instruction in instructions:
                    if(instruction == sample):
                        equal = True
                        counter = counter + 1
                if(counter>max_duplicate):
                    equal = False
            counter = 0
            instructions.append(sample)

        self.widerNums = widerNums.copy()
        self.deeperNums = deeperNums.copy()
        self.states = states.copy()    
        #save the actors
        self.widerActor.save_model()
        self.deeperActor.save_model()

        return instructions

    '''
    Update the RL agents based on the validation accuracies from the sampled architectures
    instructions=None, Assuming it uses previously generated instructions 
    states=None, Assuming it uses previously generated states
    '''
    def RL_update_step(self, val_exist, val_accus, sample_size, instructions=None, states=None):
        self.prev_accu = max(val_accus)     #use best architecture validation accuracy for val_diff of next step
        if not (instructions is None):       #different instructions from the ones that are sampled, instead count the instructions
            widerNums = instructions['Wider']
            deeperNums = instructions['Deeper']
            #(widerNums, deeperNums) = self._count_instructions(instructions, sample_size)   this is the running instructions 
        else:                  #assume instructions are the same as the sampling stage
            widerNums = self.widerNums
            deeperNums = self.deeperNums

        if not (states is None):         #same states as used in sampling stage
            pass
        else:
            states = self.states

        #load models first
        self.widerActor.load_model()
        self.deeperActor.load_model()

        #normalize the reward function 
        val_accus = self.perspective_normalization(val_exist, val_accus)
        val_accus = self.statistics_normalization(val_accus)

        self.widerActor.fit(states, widerNums, val_accus)
        self.deeperActor.fit(states, deeperNums, val_accus)
        # self.show_loss(states, widerNums, deeperNums, val_accus)

        #save the actors
        self.widerActor.save_model()
        self.deeperActor.save_model()

    '''
    predict based on the given state, give insight into the knowledge of the agent
    states=None, Assuming it uses previously generated states
    '''
    def RL_status(self, states=None):
        if not (states is None):
            pass
        else:
            states = self.states
        print ("probability predicted by the wider actor is:")
        predictions = self.widerActor.predict(states)
        print (predictions[0])
        print ("probability predicted by the deeper actor is:")
        predictions = self.deeperActor.predict(states)
        print (predictions[0])

    '''
    gather information from instructions (used by RL_update_step)
    '''
    def _count_instructions(self, instructions, sample_size):
        widerNums = [0]*sample_size
        deeperNums = [0]*sample_size
        for i in range(sample_size):
            widerNums[i] = len(instructions[i]['Wider'])
            deeperNums[i] = len(instructions[i]['Deeper'])
        return (widerNums, deeperNums)


    '''
    return statistics about the architecture as input state to the actor networks 
    [val_diff, #conv, #fc]
    in a batch of size sample_size
    '''
    def state_gen(self, val_diff, architecture, sample_size, numNew):
        #x = [diffone] + [difftwo] + [float(numNew / 100)]
        accu = [val_diff]
        arch = [0,int(numNew)]
        for layer in architecture:      #count number of fc layer from the architecture
            if(layer.Ltype == 'fc'):
                arch[0] = arch[0] + 1
        return self._state_encoding(accu, arch, self.max_layers, sample_size)    #returned states as input to the decision networks

    '''
    private helper function used by state_gen
    '''
    def _state_encoding(self, accu, arch, max_size, sample_size):
        #with thermometer_encoding
        # encoded = thermometer_encoding(arch, max_size)
        # x = accu + encoded
        # states = x.copy()
        # for i in range(sample_size-1):      #repeat the state x times 
        #     states.extend(x)
        # states = np.asarray(states).reshape(sample_size,-1)

        arch1 = decimal_encoding(arch[0], max_size)
        numNew = decimal_encoding(arch[1], max_size)        #put it in the same scale as arch
        x = accu + [arch1] + [numNew]
        states = x.copy()
        for i in range(sample_size-1):
            states.extend(x)            #repeat the states (batch_size) times
        states = np.asarray(states).reshape(sample_size,-1)
        return states


    '''
    randomly generate a new architecture based on the given # of wider and deeper actions 
    return an instruction for a new architecture transformation 
    '''
    def _random_sample(self, architecture, num_wider, num_deeper):
        architecture = copy.deepcopy(architecture)  
        max_len = len(architecture) - 1
        Wider = []
        Deeper = []

        for i in range(num_deeper):
            '''
            Always apply deeperNet first
            '''
            layer_num = random.randint(0, max_len)
            Deeper.append(layer_num)
            architecture.insert(layer_num+1, architecture[layer_num])
            max_len = max_len + 1

        for i in range(num_wider):
            layer_num = random.randint(0, max_len)
            Wider.append(layer_num)
        return {'Wider':Wider,'Deeper':Deeper}

    def save_actors(self):
        #save the actors
        self.widerActor.save_model()
        self.deeperActor.save_model()

    def load_actors(self):
        #load the actors
        self.widerActor.load_model()
        self.deeperActor.load_model()


    def perspective_normalization(self, Vexist, Vnew):
        Vnew[:] = [x - Vexist for x in Vnew]
        return Vnew

    def statistics_normalization(self, Vnew):
        Xmax = max(Vnew)
        Xmin = min(Vnew)
        if (Xmax != Xmin):
            Vnew[:] = [(2*((x-Xmin)/(Xmax-Xmin))-1) for x in Vnew]
        return Vnew

    #show the values that are calculated in the training function for DeeperActor and WiderActor
    def show_loss(self, states, widerNums, deeperNums, val_accus):

        print ("widerNums are:")
        print (widerNums)
        print ("deeperNums are:")
        print (deeperNums)

        print ("show values in training function for wider actor")
        action_prob = self.widerActor.predict(states)
        #doing onehot encoding
        wider_encoding = np.zeros(action_prob.shape)
        for i in range(len(widerNums)):
            wider_encoding[i][widerNums[i]] = 1
        widerNums = wider_encoding
        
        print ("probability predicted by the wider actor is:")
        print(action_prob)

        print ("discounted reward is:")
        print(val_accus)

        action_prob = np.sum(np.multiply(action_prob, widerNums), axis=1)
        print ("action * prob is:")
        print (action_prob)

        log_action_prob = np.log(action_prob)
        print ("action prob log is")
        print (log_action_prob)

        loss = -np.multiply(log_action_prob, val_accus)
        loss = np.mean(loss)
        print ("loss is:")
        print (loss)

        entropy = -np.sum(np.multiply(action_prob,log_action_prob))
        print ("entropy is:")
        print (entropy)

        loss = loss - self.entropy_penalty * entropy
        print ("loss with entropy is:")
        print (loss)
        print ("-------------------------------------------------------------")
        print ("-------------------------------------------------------------")

        print ("show values in training function for deeperActor actor")
        action_prob = self.deeperActor.predict(states)
        deeper_encoding = np.zeros(action_prob.shape)
        for i in range(len(deeperNums)):
            deeper_encoding[i][deeperNums[i]] = 1
        deeperNums = deeper_encoding
        
        print ("probability predicted by the deeper actor is:")
        print(action_prob)

        print ("discounted reward is:")
        print(val_accus)

        action_prob = np.sum(np.multiply(action_prob,deeperNums), axis=1)
        print ("action * prob is:")
        print (action_prob)

        log_action_prob = np.log(action_prob)
        print ("action prob log is")
        print (log_action_prob)

        loss = -np.multiply(log_action_prob, val_accus)
        loss = np.mean(loss)
        print ("loss is:")
        print (loss)

        entropy = -np.sum(np.multiply(action_prob, log_action_prob))
        print ("entropy is:")
        print (entropy)

        loss = loss - self.entropy_penalty * entropy
        print ("loss with entropy is:")
        print (loss)









