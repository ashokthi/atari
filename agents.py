import tensorflow as tf
import numpy as np

#if you mess up terminal states in sarsa you get exploding gradients?
#look here: https://web.stanford.edu/class/cs221/2017/restricted/p-final/vishakh/final.pdf
#apparently, high learning rate leads to nan gradients, exploding q-values? - keep alpha at 10^-6?
"""
The behaviour policy during training was e-greedy with e annealed linearly from 1.0 to 0.1 over the first million frames, 
and fixed at 0.1 thereafter. We trained for a total of 50 million frames (that is, around 38 days of game experience in total) 
and used a replay memory of 1 million most recent frames.
Following previous approaches to playing Atari 2600 games, we also use a simple frame-skipping technique15. 
More precisely, the agent sees and selects actions on every kth frame instead of every frame, and its last action 
is repeated on skipped frames. Because running the emulator forward for one step requires much less computation 
than having the agent select an action, this technique allows the agent to play roughly k times more games without 
significantly increasing the runtime. We use k 5 4 for all games.
"""
#Above from paper - Human level control from DQL

#Perhaps try A3c, much faster training?

#use TensorBoard for graphs - look again at tutorial for TensorBoard usage
#check if biases are present and their initialization
#change so episode ends after losing one life - how?
#try Q-learning later as in DeepMind Atari paper
#Use DeepMind architecture and preprocessing? Transform to grayscale, crop to 84x84, take last four frames as input, etc
#Try to ensure you are using GPU (get code from starcraft agent?) - GPU needs square input apparently - check this
#Use DeepMind formulation of q-values, takes observation as input then outputs q-values for all actions; doesnt require the
#forward pass - think about this; need masks and so on to do this - think
#think about implementing experience replay across many episodes then training a random sample minibatch after several episodes
#breaks correlation and reduces variance
#get nan values for loss because gradients explode; aggressive gradient clipping can work; try better optimizer maybe also?

class Sarsa(object): #standard q(s,a) method, then array for deep q learning
    def __init__(self):
        self.epsilon = 0.6

    def network(self,obs,actions): #must add actions to input; where? after conv obs, add to dense layers
        conv1 = tf.layers.conv2d(obs,16,8,4,'same','channels_last',activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1,32,4,2, 'same', 'channels_last',activation=tf.nn.relu)
        hidden = tf.layers.dense(tf.concat([tf.layers.flatten(conv2),actions],1),256,activation=tf.nn.relu) #must flatten output of conv before dense layer
        output = tf.layers.dense(hidden,1,activation=None)
        return output

    def step(self,obs,sess):
        if np.random.rand() < self.epsilon - 0.58*(tf.train.global_step(sess,self.global_step)/1000.0):
            action = np.random.choice([0,1,2,3])
        else:
            action_value = np.empty(4)
            for i in range(4):
                action_value[i] = sess.run(self.value,{self.obs_ph:np.expand_dims(obs,0),self.actions_ph:np.expand_dims([i],0)})
            action = np.argmax(action_value)
        return action


    def preprocess(self,raw_obs): #should this be part of tensorgraph since its a tf function? use np instead; not applied to batch
        #transform to grayscale and crop at the very least?
        raw_obs = np.reshape(raw_obs,[210,160,3,4])
        obs = np.mean(raw_obs,2) #the tf.to_grayscale can handle batch obs
        obs = np.squeeze(obs) #drop dimension size 1
        obs = obs[::2,::2,:] #downsamples by taking every second element in width and height pixel
        return obs

    def calcReturns(self,rewards,next_obs,next_actions,is_done,disc,sess):
        #have you corrected this for terminal episodes - very episode for Sarsa?
        #need to include done vector as a mask? and remove value for terminal state?
        returns = rewards + disc*sess.run(self.value,{self.obs_ph:next_obs,self.actions_ph:next_actions})
        returns[is_done] = rewards[is_done]
        return returns



    def build(self,sess):
        self.actions_ph = tf.placeholder(tf.float16,[None,1])
        self.obs_ph = tf.placeholder(tf.float16,[None,105,80,4])
        self.returns_ph = tf.placeholder(tf.float16,[None,1])
        self.value = self.network(self.obs_ph,self.actions_ph)
        self.loss = tf.losses.mean_squared_error(self.value,self.returns_ph)
        #consider using huber loss? apparently normalises loss? read up on this
        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(0.05)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        #self.train_op = optimizer.minimize(self.loss,self.global_step)

        init = tf.global_variables_initializer()
        sess.run(init)


    def train(self,sess,feed):
        _,loss = sess.run([self.train_op,self.loss],{self.returns_ph:feed[0],self.obs_ph:feed[1],self.actions_ph:feed[2]})
        print(loss)