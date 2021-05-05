import numpy as np
from collections import defaultdict
import random
# !pip3 install git+https://github.com/slremy/netsapi --user --upgrade
from netsapi.challenge import *

os.environ["http_proxy"] = "http://127.0.0.1:1080/"
os.environ["https_proxy"] = "http://127.0.0.1:1080/"

class Adjust01(object):
    def __init__(self,env):
        self.env = env
        self.best_policy = None
        self.best_reward = -1e9
        self.policies = []
        self.rewards = []

    def encode(self, p):
        policy = {i+1: [float(p[i*2]), float(p[i*2+1])] for i in range(5)}
        return policy

    # note that even if we only consider 01 values, the action space is still 2^10=1024
    # which is still impossible for an unstructured explore strategy within 20 epoches.
    # however, if we constrain action1 = 1 - action2, the action space will be 2^5=32 within 20 epoches
    # I think setting action1 = 1 - action2 should be treated as cheating
    # (it's indeed better, because there maybe a few good policies with in these 32, and they are nearly guaranteed to be found),
    # so I didn't use that.
    def choose_policy(self):
        if len(self.policies) < 10:
            p = np.random.randint(0, 2, size=(10,), dtype=np.int32)
            policy = self.encode(p)
        else:
            p = np.zeros(10, dtype=np.int32)
            for i in range(10):
                min_r = np.min(np.array(self.rewards))
                sums = np.zeros((2,))
                for j in range(len(self.policies)):
                    sums[self.policies[j][i]] += self.rewards[j] - min_r
                p0 = sums[0]/(sums[1] + sums[0])
                rp = np.random.uniform()
                if rp < p0:
                    p[i] = 0
                else:
                    p[i] = 1
            # p = np.clip(p, 0, 1)
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
            print(p, r)
            self.update(p, r)

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        self.train()
        best_policy = self.best_policy
        best_reward = self.env.evaluatePolicy(best_policy)

        print(best_policy, best_reward)

        return best_policy, best_reward

EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, Adjust01, "Adjust01_submission.csv")
EvaluateChallengeSubmission(ChallengeProveEnvironment, Adjust01, "Adjust01_submission.csv")
