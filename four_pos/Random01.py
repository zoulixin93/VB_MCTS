import numpy as np
from collections import defaultdict
import random
# !pip3 install git+https://github.com/slremy/netsapi --user --upgrade
# from netsapi.challenge import *
from environment.challenge import *

os.environ["http_proxy"] = "http://127.0.0.1:1080/"
os.environ["https_proxy"] = "http://127.0.0.1:1080/"

class RandomBaseline(object):
    def __init__(self,env):
        self.env = env
        self.best_policy = None
        self.best_reward = -1e9

    def choose_policy(self):
        p = np.random.randint(0, 2, size=(10))
        policy = {i+1: [float(p[i*2]), float(p[i*2+1])] for i in range(5)}
        return policy

    def update(self, policy, reward):
        if reward > self.best_reward:
            self.best_policy = policy
            self.best_reward = reward

    def train(self):
        for _ in range(20): #Do not change
            self.env.reset()
            policy = self.choose_policy()
            r = self.env.evaluatePolicy(policy)
            self.update(policy, r)

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        self.train()
        best_policy = self.best_policy
        best_reward = self.env.evaluatePolicy(best_policy)

        print(best_policy, best_reward)

        return best_policy, best_reward

EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, RandomBaseline, "RB01_submission.csv")
