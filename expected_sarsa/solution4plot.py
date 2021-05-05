import random
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from sys import exit, exc_info, argv
from IPython.display import clear_output
# from netsapi.challenge import *
from environment.challenge import *

class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):

        possible_acts = {}
        possible_acts[1] = [item for sublist in [[(x, y) for x in [0]] for y in [
            0.6 + 0.03 * i for i in range(11)]] for item in sublist]
        possible_acts[2] = [item for sublist in [
            [(x, y) for x in [0.6 + 0.03 * i for i in range(11)]] for y in [0]] for item in sublist]
        possible_acts[3] = [item for sublist in [[(x, y) for x in [0]] for y in [
            0.6 + 0.03 * i for i in range(11)]] for item in sublist]
        possible_acts[4] = [item for sublist in [
            [(x, y) for x in [0.6 + 0.03 * i for i in range(11)]] for y in [0]] for item in sublist]
        possible_acts[5] = [item for sublist in [[(x, y) for x in [0]] for y in [
            0.6 + 0.03 * i for i in range(11)]] for item in sublist]
        possible_acts[6] = [item for sublist in [[(x, y) for x in [0, 0.6]] for y in [
            0.1 * i for i in range(11)]] for item in sublist]

        self.dict = {x: possible_acts[x] for x in range(1, 7)}
        self.get_legal_actions = lambda x: self.dict[x]
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value

    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0
        value = np.max(np.array([self.get_qvalue(state, action)
                                 for action in possible_actions]))
        return value

    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha
        qvalue = (1 - learning_rate) * self.get_qvalue(state, action)\
            + learning_rate * (reward + gamma * self.get_value(next_state))
        self.set_qvalue(state, action, qvalue)

    def update_2_step(self, state, action, reward, next_state,
                      next_reward, next_next_state):
        gamma = self.discount
        learning_rate = self.alpha
        qvalue = (1 - learning_rate) * self.get_qvalue(state, action) +\
            learning_rate * (reward + gamma * next_reward +
                             gamma ** 2 * self.get_value(next_next_state))
        self.set_qvalue(state, action, qvalue)

    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        best_qvalue = np.max([self.get_qvalue(state, action)
                              for action in possible_actions])
        options = [x for x in possible_actions if self.get_qvalue(
            state, x) == best_qvalue]
        choice = np.random.choice(list(range(len(options))))
        best_action = options[choice]
        return best_action

    def get_action(self, state):
        possible_actions = self.get_legal_actions(state)
        action = None
        if len(possible_actions) == 0:
            return None
        epsilon = self.epsilon
        random_uniform = np.random.uniform(0, 1)

        if random_uniform < epsilon:
            chosen_action = random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action


class EVSarsaAgent(QLearningAgent):
    def get_value(self, state):
        epsilon = self.epsilon
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return 0.0

        values = [self.get_qvalue(state, action)
                  for action in possible_actions]

        best_action = np.argmax(values)

        state_value = sum([self.eval_prob(best_action, action, len(possible_actions), epsilon) *
                           self.get_qvalue(state, action) for action in possible_actions])
        return state_value

    def eval_prob(self, best_action, action, num_actions, eps):
        if (action == best_action).all():
            return 1 - eps + eps/num_actions
        else:
            return eps/num_actions


class CustomAgent:
    def __init__(self, environment, head):
        self.environment = environment
        self.episode_number = 20

        self.run = []
        self.scores = []
        self.policies = []
        self.f = open(head + "_" + str(np.random.choice(range(100000), (1,))[0]) + "_res.txt", "w")

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        rewards = []
        try:
            # select the policies learnt in this case we are simply randomly generating
            agent = EVSarsaAgent(alpha=0.5, epsilon=0., discount=0.99,
                                 get_legal_actions=lambda s: range(3))
            for i in range(self.episode_number):
                self.environment.reset()
                if i <= 13:
                    agent.epsilon = 1.0
                else:
                    agent.epsilon = 0
                policy = {}
                episodic_reward = 0
                for j in range(5):
                    action = agent.get_action(self.environment.state)
                    #self.environment.policy[str(self.environment.state)] = action

                    prev_state = self.environment.state
                    ob, reward, done, _ = self.environment.evaluateAction(
                        action)
                    agent.update(prev_state, action, reward, ob)
                    episodic_reward += reward
                    policy[str(j+1)] = action
                rewards.append(episodic_reward)
                candidates.append(policy)
                self.f.writelines(str(policy)+"\t"+str(episodic_reward)+"\n")
                self.f.flush()

            #rewards = self.environment.evaluatePolicy(candidates)
            best_policy = candidates[np.argmax(rewards)]
            best_reward = rewards[np.argmax(rewards)]
            print(best_reward, best_policy)
        except (KeyboardInterrupt, SystemExit):
            print(exc_info())

        return best_policy, best_reward

    def scoringFunction(self):
        scores = []
        for ii in range(10):
            self.environment.reset()
            finalresult, reward = self.generate()
            self.policies.append(finalresult)
            self.scores.append(reward)
            self.run.append(ii)

        return np.mean(self.scores)/np.std(self.scores)

    def create_submissions(self, filename='my_submission.csv'):
        labels = ['run', 'reward', 'policy']
        rewards = np.array(self.scores)
        data = {'run': self.run,
                'rewards': rewards,
                'policy': self.policies,
                }
        submission_file = pd.DataFrame(data)
        submission_file.to_csv(filename, index=False)

if __name__ == '__main__':
    for _ in range(10):
        env = ChallengeSeqDecEnvironment(515)
        CustomAgent(env,"seque").generate()
    for _ in range(10):
        env = ChallengeProveEnvironment(515)
        CustomAgent(env, "prove").generate()


