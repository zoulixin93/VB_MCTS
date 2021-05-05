import numpy as np
from collections import defaultdict
import random
# !pip3 install git+https://github.com/slremy/netsapi --user --upgrade
# from netsapi.challenge import *
from environment.challenge import *
# os.environ["http_proxy"] = "http://127.0.0.1:1081/"
# os.environ["https_proxy"] = "http://127.0.0.1:1081/"

class CEM(object):
    def __init__(self,env):
        self.env = env
        self.best_policy = None
        self.best_reward = -1e9
        self.policies = []
        self.rewards = []

    def choose_policy1m0(self):
        x = np.zeros((10))
        for i in range(5):
            r = np.random.uniform(0, 1)
            x[2*i] = r
            x[2*i+1] = 1 - r
        policy = {i+1: [float(x[i*2]), float(x[i*2+1])] for i in range(5)}
        return x, policy

    def choose_policy_uniform(self):
        x = np.random.uniform(size=(10))
        policy = {i+1: [float(x[i*2]), float(x[i*2+1])] for i in range(5)}
        return x, policy

    def encode(self, p):
        policy = {i+1: [float(p[i*2]), float(p[i*2+1])] for i in range(5)}
        return policy

    def choose_policy(self):
        if len(self.policies) < 10:
            # p = np.random.randint(0, 2, size=(10))
            # p, policy = self.choose_policy1m0()
            p = np.random.uniform(size=(10,))
            policy = self.encode(p)
        else:
            idx = np.argsort(np.array(self.rewards))
            hp = np.array(self.policies)[idx[-3:]]
            hr = np.array(self.rewards)[idx[-3:]]
            p = np.zeros((10,))
            print(self.rewards)
            print(hr)
            for i in range(10):
                mean = np.mean(hp[:, i])
                std = np.std(hp[:, i])
                p[i] = np.random.normal(mean, std)
                print(std)
            p = np.clip(p, 0, 1)
            policy = self.encode(p)
        return p, policy

    def update(self, p, reward):
        if reward > self.best_reward:
            self.best_policy = self.encode(p)
            self.best_reward = reward
        self.policies.append(p)
        self.rewards.append(reward)

    def train(self):
        for _ in range(20): #Do not change
            self.env.reset()
            p, policy = self.choose_policy()
            r = self.env.evaluatePolicy(policy)
            self.update(p, r)

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        self.train()
        best_policy = self.best_policy
        best_reward = self.env.evaluatePolicy(best_policy)

        print(best_policy, best_reward)

        return best_policy, best_reward

EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CEM, "CEM4seqsubmission.csv")
EvaluateChallengeSubmission(ChallengeProveEnvironment, CEM, "CEM4prosubmission.csv")
