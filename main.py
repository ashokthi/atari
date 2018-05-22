import gym
import tensorflow as tf
import numpy as np
import random
from agents import Sarsa

env = gym.make("Breakout-v0")
agent = Sarsa()
sess = tf.Session()
agent.build(sess)

disc = 0.95
rewardsAll = []
actionsAll = []
obsAll = []
obsproc = []
doneList = []

for i_episode in range(100):
    print(i_episode)
    obs = env.reset()
    obsAll.append(obs)


    for t in range(500):
        env.render()
        if t<3:
            action = 0 #env.action_space.sample()
            #print(action)
        else:
            action = agent.step(obsproc[-1],sess)
        #might need to modify this to account for done / terminal states
        obs,reward,done,_ = env.step(action)
        rewardsAll.append(reward)
        actionsAll.append(action)
        obsAll.append(obs)
        doneList.append(done)

        if t>=2:
            #print(len(obsAll))
            obsproc.append(agent.preprocess(obsAll[-4:]))

        if len(obsproc)+1 % 10000 == 0:
            rewardsAll = []
            actionsAll = []
            obsAll = []
            obsproc = []
            doneList = []

        if (t+1) % 45 == 0:
            print("train")
            # sample indices from len obsproc then draw those samples (with indices appropriately shifted) from all lists
            #create next_obs, next_actions and put into batch
            #calculate returns
            indices = np.random.choice(range(len(obsproc)-3),32,replace=False)
            rewards = np.array(rewardsAll)[indices+3]
            rewards = np.reshape(rewards,[32,1])
            obsTr = np.reshape(np.array(obsproc)[indices],[32,105,80,4])
            nextObsTr = np.reshape(np.array(obsproc)[indices+1],[32,105,80,4])
            actions = np.reshape(np.array(actionsAll)[indices+3],[32,1])
            nextActions = np.reshape(np.array(actionsAll)[indices+4],[32,1])
            is_done = np.reshape(np.array(doneList)[indices+3],[32,1])
            returns = agent.calcReturns(rewards, nextObsTr, nextActions, is_done, disc, sess)
            agent.train(sess,feed=[returns,obsTr,actions])

        if done:
            obsAll = obsAll[:-1]
            print("done")
            break
