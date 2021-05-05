import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
from sys import exit, exc_info, argv
import random
import pandas as pd

# os.environ["http_proxy"] = "http://127.0.0.1:1080/"
# os.environ["https_proxy"] = "http://127.0.0.1:1080/"

# from netsapi.challenge import *
from environment.challenge import *

try:
    xrange = xrange
except:
    xrange = range

gamma = 0.99

env = ChallengeProveEnvironment()

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

def actionSpace(resolution):
    x,y = np.meshgrid(np.arange(0,1+resolution,resolution), np.arange(0,1+resolution,resolution))
    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
    return xy.round(2).tolist()

action_resolution = 0.2

actions = actionSpace(action_resolution)
a_s = len(actions)
actionspace = range(a_s-1)


tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=1e-2,s_size=1,a_size=a_s,h_size=8) #Load the agent.

total_episodes = 20 #Set total number of episodes to train agent on.
max_ep = 5
update_frequency = 4

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []
        
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    while i < total_episodes:
        env.reset()
        s1 = env.state
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            s = s1
            s = np.expand_dims(s, axis=0)
            print('s=',s)
            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)
            env_action = actions[a] #convert to ITN/IRS
            print('env_action', env_action)
            s1,r,d,_  = env.evaluateAction(env_action)
            ep_history.append([s,a,r,s1])

            running_reward += r
            if d == True:
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={myAgent.reward_holder:ep_history[:,2],
                        myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                
                total_reward.append(running_reward)
                total_length.append(j)
                break
        i += 1
                
    #Now extract final policy
    best_policy = {}
    for j in range(5):

        s = j+1
        s = np.expand_dims(s, axis=0)

        #Probabilistically pick an action given our network outputs.
        a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
        a = np.random.choice(a_dist[0],p=a_dist[0])
        a = np.argmax(a_dist == a)
        env_action = actions[a] #convert to ITN/IRS
        best_policy[int(s)]= env_action
        
    print('best_policy',best_policy)
    best_reward = env.evaluatePolicy(best_policy)
    print('best_reward',best_policy,best_reward)
