from sys import exit, exc_info, argv
import numpy as np
import pandas as pd
from environment.challenge import *
# from netsapi.challenge import *


class CustomAgent:
    def __init__(self, environment,head):
        self.f = open(head + "_" + str(np.random.choice(range(100000), (1,))[0]) + "_res.txt", "w")
        self.environment = environment

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        rewards = []
        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            for i in range(20):
                self.environment.reset()
                policy = {}
                for j in range(5):  # episode length
                    policy[str(j + 1)] = [random.random(), random.random()]
                candidates.append(policy)
                rewards.append(self.environment.evaluatePolicy(policy))
                tt_p = candidates[np.argmax(rewards)]
                tt_r = rewards[np.argmax(rewards)]
                print(tt_p,tt_r)
                self.f.writelines(str(tt_p) + "\t" + str(tt_r) + "\n")
                self.f.flush()
            best_policy = candidates[np.argmax(rewards)]
            best_reward = rewards[np.argmax(rewards)]
            print(best_policy, best_reward)

        except (KeyboardInterrupt, SystemExit):
            print(exc_info())

        return best_policy, best_reward

if __name__ == '__main__':
    for _ in range(10):
        env = ChallengeSeqDecEnvironment(515)
        CustomAgent(env,"seque").generate()
    for _ in range(10):
        env = ChallengeProveEnvironment(515)
        CustomAgent(env, "prove").generate()