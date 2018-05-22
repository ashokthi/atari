import tensorflow as tf
import numpy as np


class Sarsa(object):
    def __init__(self):
        self.epsilon = 0.6

    def network(self,obs,actions):
        conv1 = tf.layers.conv2d(obs,16,8,4,'same','channels_last',activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1,32,4,2, 'same', 'channels_last',activation=tf.nn.relu)
        hidden = tf.layers.dense(tf.concat([tf.layers.flatten(conv2),actions],1),256,activation=tf.nn.relu)
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


    def preprocess(self,raw_obs):
        raw_obs = np.reshape(raw_obs,[210,160,3,4])
        obs = np.mean(raw_obs,2)
        obs = np.squeeze(obs)
        obs = obs[::2,::2,:]
        return obs

    def calcReturns(self,rewards,next_obs,next_actions,is_done,disc,sess):
        returns = rewards + disc*sess.run(self.value,{self.obs_ph:next_obs,self.actions_ph:next_actions})
        returns[is_done] = rewards[is_done]
        return returns



    def build(self,sess):
        self.actions_ph = tf.placeholder(tf.float16,[None,1])
        self.obs_ph = tf.placeholder(tf.float16,[None,105,80,4])
        self.returns_ph = tf.placeholder(tf.float16,[None,1])
        self.value = self.network(self.obs_ph,self.actions_ph)
        self.loss = tf.losses.mean_squared_error(self.value,self.returns_ph)
        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(0.05)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)

        init = tf.global_variables_initializer()
        sess.run(init)


    def train(self,sess,feed):
        _,loss = sess.run([self.train_op,self.loss],{self.returns_ph:feed[0],self.obs_ph:feed[1],self.actions_ph:feed[2]})
        print(loss)