from sys import exit, exc_info, argv
import numpy as np
import pandas as pd

from netsapi.challenge import *


class CustomAgent:
    def __init__(self, environment):
        self.environment = environment

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            for i in range(20):
                self.environment.reset()
                policy = {}
                for j in range(5):  # episode length
                    policy[str(j + 1)] = [random.random(), random.random()]
                candidates.append(policy)

            rewards = self.environment.evaluatePolicy(candidates)
            best_policy = candidates[np.argmax(rewards)]
            best_reward = rewards[np.argmax(rewards)]
            print(best_policy, best_reward)

        except (KeyboardInterrupt, SystemExit):
            print(exc_info())

        return best_policy, best_reward
if __name__ == '__main__':
    EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CustomAgent, "res1.csv")
    EvaluateChallengeSubmission(ChallengeProveEnvironment, CustomAgent, "res2.csv")