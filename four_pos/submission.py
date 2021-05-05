from sklearn.linear_model import LinearRegression
import numpy as np
import random
from collections import defaultdict
import random
# from netsapi.challenge import *
from environment.challenge import *

class LinearForest:
    def __init__(self,
                 feature_num=4,
                 model_num=100,
                 survive_num=5,
                 input_size=10):
        self.feature_num = feature_num
        self.model_num = model_num
        self.survive_num = survive_num
        self.input_size = input_size

    def fit(self, X, y):
        self.models = []
        self.scores = []
        self.selects = []
        self.select_model_index = None
        self.models_set = set()
        for i in range(self.model_num):
            select = sorted(random.choices(list(range(self.input_size)), k=self.feature_num))
            if str(select) in self.models_set:
                continue
            self.models_set.add(str(select))
            self.selects.append(select)
            t_X = X[:, select]
            t_model = LinearRegression().fit(t_X, y)
            self.models.append(t_model)
            self.scores.append(t_model.score(t_X, y))
        self.select_model_index = np.argsort(self.scores)[-self.survive_num:]

    def predict(self, X):
        y = np.zeros((X.shape[0]))
        for idx in self.select_model_index:
            # print(idx)
            # print(self.scores[idx])
            t_X = X[:, self.selects[idx]]
            # print(self.selects[idx])
            y += self.models[idx].predict(t_X)
        y /= len(self.select_model_index)
        return y

    def diff(self, X, y):
        pred_y = self.predict(X)
        d = np.sqrt(np.mean(np.square(pred_y - y)))
        return d

class CustomAgent(object):
    def __init__(self, env):
        self.env = env
        self.best_policy = None
        self.best_p = None
        self.best_reward = -1e9
        self.sample_num = 10
        self.warm_up = 8
        self.ps = []
        self.rs = []

    def encode(self, p):
        policy = {i+1: [float(p[i*2]), float(p[i*2+1])] for i in range(5)}
        return policy

    def choose_policy_lf(self):
        best_pred = -1e9
        best_p = None
        model = LinearForest()
        model.fit(np.array(self.ps), np.array(self.rs))
        print("model diff {}".format(model.diff(np.array(self.ps), np.array(self.rs))))
        for i in range(self.sample_num):
            p = np.random.uniform(0, 1, size=(1, 10))
            p_score = model.predict(p)
            p = np.reshape(p, (-1))
            if p_score > best_pred:
                policy = self.encode(p)
                best_pred = p_score
                best_p = p
        print("model rew {}".format(best_pred))
        return p, policy

    def choose_policy_s(self):
        best_pred = -1e9
        best_p = None
        model = LinearForest(survive_num=3, feature_num=3, model_num=100)
        model.fit(np.array(self.ps), np.array(self.rs))
        select = model.selects[model.select_model_index[0]]
        print(select)
        print("model diff {}".format(model.diff(np.array(self.ps), np.array(self.rs))))
        for i in range(self.sample_num):
            p = np.random.uniform(0, 1, size=(1, 10))
            p[0, select] = self.best_p[select]
            p_score = model.predict(p)
            p = np.reshape(p, (-1))
            if p_score > best_pred:
                policy = self.encode(p)
                best_pred = p_score
                best_p = p
        print("model rew {}".format(best_pred))
        return p, policy

    def choose_policy_uniform(self):
        p = np.random.uniform(size=(10))
        return p, self.encode(p)

    def update(self, p, policy, reward):
        if reward > self.best_reward:
            self.best_policy = policy
            self.best_reward = reward
            self.best_p = p
        self.ps.append(p)
        self.rs.append(reward)

    def train(self):
        for _ in range(20): #Do not change
            self.env.reset()
            if _ < self.warm_up:
                p, policy = self.choose_policy_uniform()
            else:
                p, policy = self.choose_policy_s()
            r = self.env.evaluatePolicy(policy)
            print(r, policy)
            self.update(p, policy, r)

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        self.train()
        best_policy = self.best_policy
        best_reward = self.env.evaluatePolicy(best_policy)

        print(best_policy, best_reward)

        return best_policy, best_reward

if __name__ == "__main__":
    # test()
    EvaluateChallengeSubmission(ChallengeProveEnvironment, CustomAgent, "seqsub.csv")
    EvaluateChallengeSubmission(ChallengeProveEnvironment, CustomAgent, "prosub.csv")
