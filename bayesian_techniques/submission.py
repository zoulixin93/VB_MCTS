'''
KDD 2019 | Policy Learning for Malaria Control
https://compete.hexagon-ml.com/practice/rl_competition/37/

Participants are expected to submit high performing solutions to the sequential decision making task **under small number of episodes**.

We develop the data efficient algorithm to train the agent for the sequential decision making task under small number of episodes.

----
# Papers
- Infomax strategies for an optimal balance between exploration and exploitation
    - https://arxiv.org/abs/1601.03073
- The End of Optimism? An Asymptotic Analysis of Finite-Armed Linear Bandits
    - https://arxiv.org/abs/1610.04491
    - Warmup phase => Success phase => Recovery phase
- Sequential Learning under Probabilistic Constraints
    - http://auai.org/uai2018/proceedings/papers/233.pdf

----
#
# Concept of algorithm
#

Our algorithm aims to maximize the rewards at each time step (year) instead of episode.  
We approximate the optimisation function as the one at each year,  
    maximize R^* = \sum_t reward_t
    => 
    r_t^* = maximize {reward_t}, t = 1, ..., 5
    R = \sum_t r_t^*

We use the previous action as the current state because the state information cannot be acquired enough and we can't train the agent enough due to the expensive computational cost.
    r_t^* = maximize_{action_t} {reward_t|state=action_t, action_{t-1}}, t = 2, ..., 5
Action at year 1 is selected by Baysian Optimization and epsilon-search.


#
# Method
#

1. Exploration of action and learning of environment throughout years
    - Thompson Sampling
        - Explore environment and exploit knowledge

2. Decision making
    - year1: the action of maximum reward in the history
    - year2-5:
        - Gaussian Process fitting with 20 episode data
            - r_t = f(a_t|a_{t-1})
            - action_t = argmax_a f(a|action_{t-1})
            - Shared environment learning
    - Our GP don't input the action sequence through year1 to year5 because we can't get much samples and GP is not good at handling many features.

3. Find best policy with rolling optimization
    - action_i = \argmax_{a_i} \sum_{t=i}^{5-r} f_gp (a_t|a_{t-1},...,a_1)
    - given, a_{i-1}, ..., a_1
    - a \in A
    - r \in 5 - i, ..., 1 (parameter, rolling years)
    - This script select 3 years of rolling
    - Recursive search

3. Discretize action space (The action space should be distributed at equal intervals to cope with the new environment)
    - Action resolution is grid shape.

4. We make simulator, which is similar to the original environment, to evaluate our algorithms.

5. It is not used in the case of the sampling whose **consecutive sequence** was a negative reward in the past
    - (1, 0) => (1, 0): -10 :=> not sample
    - (1, 0) => (1, 0): np.mean([30, 30]) < past mean :=> not sample

6. Tuning of GP kernel is important
    - length scale bounds is stable with "fix"
    - also adjust the length scale to the "y" scale

7. Epsilon-Gaussia-Process with Thompthon Sampling
    ```
    if episode < 10 or 1 - e <= threshold
        a <- TS_t(a, b)
    else:  # episode >= 10 and random.random() > explore_threshold
        # GP: a_{t-1}, a_{t} => r
        Fitting GP from the history so far
        a <- random pick up (\argmax_top10 mu(a_{t-1}, a_t))
            where a_{t-1} = previous action
    ```

8. Tuning the discretized action space for exploration => 0.25 intervals
    - The input of GP to fit at the end of episodes is transformed into fine-grained intervals
    - The input round of GP to fit at the end of episodes was set to 3

9. Affect the update of alpha and beta of TS around the target action position.
    - Update the values in the radius: np.sqrt(action_resolution**2 + action_resolution**2)
    - Change of update varies according to the distance

10. Search action of year 1 with Bayesian optimization under Upper Confidence Bound
    - The fitting at year1 GP is risky
    - => Bayesian Optimization
    - Search wisely and choose max value action in the observations
        - Action decision at year1 is important
        - Another logic is used in the action decision at year1
    2. NOTE: fitting all actions to choose the action at year1

11. The change of update is calculated by sigmoid-like custom function
    - abs(x) <= 1: x
    - otherwise: np.sign(x) * (1 + np.log(x))
    - NOTE: hypabolic tangent (cannot handle a large reward)

13. [FAILED] Use year feature (=state) in the GP fitting
    - 1 => 0.00, 2 => 0.25, 3 => 0.50, 4 => 0.75, 5 =>1.00

14. Tuning kappa, UCB parameter: 100 => 200


==============================================
[IDEA NOTEs]
- Scaling of rewards
- Adjustment with mean and standard deviation
- Algorithm with accumulated reward
- Improve thompthon sampling
    - Current: the information and distributions are shared between time steps
    - Idea1: create and learn adjustment term at each time step
- Improve the search by using GP
- Efficient exploration algorithm
- Train models with three or more consecutive relationships of actions
- Multi agent for robustness
- Use rewards for state
- Normalization of reward
- Meta learner
    - Learn a controller that controls time-dependent sampling probability in TS
- Increase training data with history trajectories
- Adaptive search space
- Predict reward with shared GP and GP at each year
    - reward(t) = (1 - c) * GP_share(a_{t-1}, a_t) + c * GP_t(a_t) # + GP(a_t|t=1,...,5)

==============================================
'''
from collections import ChainMap
import copy
import numpy as np
import pandas as pd
import random
from scipy.stats import norm

from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

# Libraly for simulation environment
# !pip3 install git+https://github.com/slremy/netsapi --user --upgrade
from netsapi.challenge import EvaluateChallengeSubmission, ChallengeSeqDecEnvironment, ChallengeProveEnvironment


#
# Algorithms
#

class BayesianOptimization():
    '''
    >>> exploration_space = [(round(x, 2), round(y, 2)) for x in np.arange(0, 1.1, 0.1) for y in np.arange(0, 1.1, 0.1)]
    >>> function_value = [random.randint(0, 30) for _ in range(len(exploration_space))]
    >>> function_value = _generate_gaussian_reward_map()
    >>> utility_function = 'ucb'
    >>> BO = BayesianOptimization(exploration_space, utility_function)
    >>> suggestions = BO.suggest()
    >>> action = max(suggestions.items(), key=lambda x: x[1])[0]
    >>> result = function_value[int(10*action[0]), int(10*action[1])]
    >>> BO.update_history((action, result))
    >>> for iteration in range(20):
    >>>     suggestions = BO.suggest()
    >>>     action = max(suggestions.items(), key=lambda x: x[1])[0]
    >>>     print(action)
    >>>     result = function_value[int(10*action[0]), int(10*action[1])]
    >>>     BO.update_history((action, result))
    >>> print(BO.suggest())
    '''
    def __init__(self, exploration_space, utility_function):
        # Parameters
        self._exploration_space = exploration_space
        # TODO: set parameters
        self._utility_function = {
            'ucb': self._ucb,
            'ei': self._ei,
            'poi': self._poi,
        }[utility_function]

        # History
        self._history = []

    def suggest(self):
        '''Return: Dict[self._exploration_space] => utility'''
        #print('BO: suggest')
        if not self._history:
            # Random choice
            return dict((act, random.random()) for act in self._exploration_space)

        # Fit GP
        x = [xy[0] for xy in self._history]
        y = [min(xy[1], xy[1] / 10) for xy in self._history]
        #y = [xy[1] / 100 for xy in self._history]
        self._gp = GaussianProcessRegressor(
            kernel=Matern(length_scale_bounds="fixed", nu=1.5),
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        self._gp.fit(x, y)

        # Utility
        utility_value = self._utility_function(self._exploration_space, self._gp, max(y))

        return dict(zip(self._exploration_space, utility_value))

    def update_history(self, action_reward):
        #print('BO: update history')
        self._history.append(action_reward)

    def predict(self):
        if not self._history:
            return []
        x = [xy[0] for xy in self._history]
        y = [min(xy[1], xy[1] / 10) for xy in self._history]
        self._gp = GaussianProcessRegressor(
            kernel=Matern(length_scale_bounds="fixed", nu=1.5),
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        self._gp.fit(x, y)

        return list(zip(self._exploration_space, self._gp.predict(self._exploration_space)))

    def _ucb(self, x, gp, _y_max, kappa=200):
        mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    def _ei(self, x, gp, y_max, xi=0.1):
        mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    def _poi(self, x, gp, y_max, xi=0.1):
        mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return norm.cdf(z)


class CustomAgent(object):
    '''
    Custom agent
    '''
    def __init__(self, env):
        # Environment
        self.env = env

        # Parameters
        self.years = list(range(1, 6))

        # Define action space
        self.action_resolution = 0.25
        self.actions = self._make_mesh_action_space(resolution=self.action_resolution)
        self.actions = [tuple(xy_) for xy_ in self.actions]

        # Posterior parameters of Beta distribution: Alpha, Beta
        self.ActionValue = {}
        self.init = (2, 5)
        for key in self.actions:
            self.ActionValue[key] = self.init

        # Parameters to learn environment
        self.action_reward_history = {year: [] for year in self.years}

        # Parameters to explore action
        self.action_sequence = {
            action_current: {
                action_next: []
                for action_next in self.actions
            }
            for action_current in self.actions
        }

        # Bayesian Optimization for year 1
        exploration_space = self.actions
        utility_function = 'ucb'  # ucb, ei, poi
        self.BO = BayesianOptimization(exploration_space, utility_function)
        
    def choose_action(self, previous_action, episode, state):
        """Use Thompson sampling to choose action. Sample from each posterior and choose the max of the samples."""
        explore_threshold = 0.1
        # Year1: BO
        if state == 1:
            samples = self.BO.suggest()
            #print('predict: ', self.BO.predict())

        # Year2-4: TS
        elif episode < 10 or random.random() <= 1 - explore_threshold:
            #print('=== Exploration: Thompson Sampling ===')
            samples = {}
            for key in self.ActionValue:
                samples[key] = np.random.beta(self.ActionValue[key][0], self.ActionValue[key][1])
        # Year2-4: GP
        else:  # episode >= 10 and random.random() > explore_threshold
            #print('=== Exploration: Gaussian Process ===')
            # Fitting GP: NOTE: method 化しろ
            # Make training data for GPR: x = (act_{t-1}, act_t), y = (reward_t)
            a = self.action_reward_history
            x = []
            y = []
            # TODO: refactor
            is_current_terminal = False
            for ep in range(len(a[1])):  # = range(20)
                for year in self.years:
                    if year == 1:
                        continue
                    x_ = a[year-1][ep][0] + a[year][ep][0]
                    y_ = a[year][ep][1]
                    x.append(x_)
                    y.append(y_)
                    if ep + 1 == episode and year == state:
                        is_current_terminal = True
                        break
                if is_current_terminal:
                    break

            #print('=== GP I/O ===')
            #print(list(zip(x, y)))
            # Fitting GP regression
            kernel = Matern(length_scale=np.max(y)*2, length_scale_bounds="fixed", nu=1.5)  # Optimization Success
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10).fit(x, y)

            # NOTE: Action space to search.
            xy = self.actions

            # Map input to gp prediction
            input_list = [previous_action + j for j in xy]
            gp_pred = gpr.predict(input_list)  # mu: mean
            samples = {tuple(action_reward[0]): action_reward[1] for action_reward in zip(xy, gp_pred)}

            # NOTE: shuffle top actions
            top_samples_gp = 10
            sorted_samples = sorted(samples.items(), key=lambda x: x[1], reverse=True)
            top_samples = dict(sorted_samples[:top_samples_gp])
            top_samples_values = list(top_samples.values())
            top_samples_shuffle = dict(zip(top_samples, random.sample(top_samples_values, len(top_samples_values))))
            remain_samples = dict(sorted_samples[top_samples_gp:])
            samples = dict(ChainMap(top_samples_shuffle, remain_samples))

        max_value_actions = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        max_value_action =  max_value_actions[0][0]

        if state != 1:  # previous_action is not None:
            # Statistics of past actions
            past_histry = [k for i in self.action_sequence.values() for j in i.values() for k in j]
            if past_histry:
                past_mean = np.mean(past_histry)
                past_max = np.max(past_histry)
                past_min = np.min(past_histry)
            else:
                past_mean = 0
                past_max = 10000
                past_min = -10000

            # It is not used in the case of the sampling whose **consecutive sequence** was a negative reward in the past
            for a, r in max_value_actions:
                # For GP samples
                if previous_action not in self.action_sequence:  # For next action
                    self.action_sequence[previous_action] = {action_next: [] for action_next in self.actions}
                if a not in self.action_sequence[previous_action]:  # For current action
                    self.action_sequence[previous_action][a] = []

                action_sequence_ = self.action_sequence[previous_action][a]
                # If the action was not sampled in the past, use it.
                if not action_sequence_:
                    max_value_action = a
                    #print('NEW', max_value_action)
                    break
                cond_less_0 = min(action_sequence_) <= 0
                if len(action_sequence_) >= 1:  # NOTE: 2 is more robust?
                    cond_less_mean = np.mean(action_sequence_) <= past_mean
                else:
                    cond_less_mean = False

                if cond_less_0 or cond_less_mean:
                    #print(f"""
                    #    CONDITION
                    #        action_sequence_: {action_sequence_}
                    #        cond_less_0: {cond_less_0}
                    #        cond_less_mean: {cond_less_mean} {np.mean(action_sequence_)}({past_mean})
                    #        max_value_action: {max_value_action}
                    #        a: {a}
                    #        r: {r}
                    #""")
                    continue

                # If there is a sequence in the past and its reward is higher than the past average, the minimum value is not less than 0

                max_value_action = a
                #print('GOOD', max_value_action)
                break

        return max_value_action

    def huber(self, x):
        """x: reward"""
        x /= 100
        #return np.tanh(x)
        if abs(x) <= 1:
            return x
        else:
            #return np.sign(x) * x**2
            return np.sign(x) * (1 + np.log(abs(x)))

    def update(self, action, reward):
        """Update parameters of posteriors, which are Beta distributions"""
        # like puseudo count
        a, b = self.ActionValue[action]
        #print(f"UPDATE {action}: ({a}, {b})")
        a = a + self.huber(reward)  # The larger the reward, the easier it is to select
        b = b + 1 - self.huber(reward)  # It becomes easy to be selected as the reward becomes larger, and it becomes difficult to be selected as the reward becomes smaller
        a = 0.001 if a <= 0 else a
        b = 0.001 if b <= 0 else b
        
        self.ActionValue[action] = (a, b)

        #print(f"=> ({a}, {b})")

        # Update nearby action candidates
        around_update_rate = 0.3  # Parameter to adjust the degree of change according to the distance; [0, 1]
        radius = np.sqrt(self.action_resolution**2 + self.action_resolution**2 + 1e-9)  # 1e-9 is for safety to caluculate the small number 
        for action_around in self.actions:
            if action_around == action:
                continue
            x = action_around[0] - action[0]
            y = action_around[1] - action[1]
            distance = np.sqrt(x**2 + y**2)
            if distance <= radius:
                a, b = self.ActionValue[action_around]
                #print(f"UPDATE {action_around}: ({a}, {b})")
                a = a + self.huber(reward) * around_update_rate * (1 - distance)
                b = b + (1 - self.huber(reward)) * around_update_rate * (1 - distance)  # To adjust the update, weight 1-r. If normal update is 1, it will be the update of around_update_rate * (1-distance) for adjacent actions.
                a = 0.001 if a <= 0 else a
                b = 0.001 if b <= 0 else b

                #print(f"=> ({a}, {b})")

                self.ActionValue[action_around] = (a, b)

    def _make_mesh_action_space(self, resolution=0.1, round_=2):
        '''If resolution=0.1, return [[0.0, 0.0], [0.0, 0.1], ..., [1.0, 0.9], [1.0, 1.0]] (length = 121)'''
        x, y = np.meshgrid(np.arange(0, 1 + resolution, resolution), np.arange(0, 1 + resolution, resolution))
        xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        return xy.round(round_).tolist()

    def train(self):
        episode = -1  # Start from -1 because below code can't be changed
        for _ in range(20): # Do not change
            episode += 1
            # Initialize environment
            self.env.reset()
            current_state = self.env.state  # Initial state == 1
            previous_action = None

            # Training in each episode
            while True:
                # Choose action
                action =  self.choose_action(previous_action, episode, current_state)

                # Evaluate action (update environment with the action)
                next_state, reward, done, _ = self.env.evaluateAction(action)


                # TODO: NaN is generated by simulator...
                if np.isnan(reward):
                    reward = np.mean([action_reward[1] for action_reward in self.action_reward_history[current_state]])
                    reward = 30
                    #print(f"[WARNING] Reward is nan. Use {reward} instead!")

                #print(f"Action: {action}, Reward: {reward}")
                #print(self.ActionValue)

                # If current_state == 1, update BO
                if current_state == 1:
                    self.BO.update_history((action, reward))
                    #print(self.BO._history)

                # Update parameters of GP
                action_reward = [action, reward]
                self.action_reward_history[current_state].append(action_reward)

                # Update parameters of Exploration
                if previous_action is not None:
                    self.action_sequence[previous_action][action].append(reward)

                # Update parameters of TS
                # TODO: Update at current_state == 1?
                #if current_state != 1:
                #    self.update(action, reward)
                self.update(action, reward)
                current_state = next_state

                # Update previous action
                previous_action = action

                # Terminte if episode ends
                if done:
                    break

    def search_best_policy(self):
        a = self.action_reward_history

        # Bayesian Optimization of year1's action
        # Exploration: BO
        # Exploitation: greedy

        # Fitting GP regression for year2-5        
        # Make training data for GPR: x = (act_{t-1}, act_t), y = (reward_t), 2 <= t <= 5
        x = []
        y = []
        for ep in range(len(a[1])):  #range(20):  # range(len(a[1]))
            for year in self.years:
                if year == 1:
                    continue
                x_ = a[year-1][ep][0] + a[year][ep][0]
                y_ = a[year][ep][1]
                x.append(x_)
                y.append(y_)

        #print('=== GP I/O after year1 ===')
        #print(list(zip(x, y)))
        kernel = Matern(length_scale=np.max(y)*2, length_scale_bounds="fixed", nu=1.5)  # Optimization Success
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10).fit(x, y)

        # Action space to search
        xy = self._make_mesh_action_space(self.action_resolution / 2, round_=3)

        # Map input to gp prediction
        input_list = [tuple(i + j) for i in xy for j in xy]
        gp_pred = gpr.predict(input_list)
        map_input_gp_pred = dict(zip(input_list, gp_pred))

        # Find best policy
        best_policy = {year: [] for year in self.years}
        for year in self.years:
            if year == 1:
                best_policy[year] = max(a[1], key=lambda x: x[1])[0]
                #print(f"Year {year}: {best_policy[year]}")
            else:
                # Find best policy with previous action(state, observation) after second year
                #print(f"Year {year}: ")
                previous_best_action = list(best_policy[year-1])
                current_best_action_rolling = self._refine_pred(previous_best_action, xy, map_input_gp_pred, rolling_num=3)  # max(6-year, 2)
                #current_best_action_rolling = self._refine_pred(previous_best_action, xy, map_input_gp_pred, rolling_num=max(6-year, 3))  # max(6-year, 2)
                #current_best_action_rolling = self._refine_pred(previous_best_action, xy, map_input_gp_pred, rolling_num=6-year)  # max(6-year, 2)
                best_policy[year] = current_best_action_rolling[0]

        return best_policy

    def _refine_pred(self, previous_action, xy, map_input_gp_pred, rolling_num=1):
        '''
        Return best action with rolling optimization
        '''
        if rolling_num == 0:
            return None, 0
        
        reward_action = {}
        for xy_ in xy:
            max_action, max_reward = self._refine_pred(xy_, xy, map_input_gp_pred, rolling_num-1)
            reward_action[tuple(xy_)] = map_input_gp_pred[tuple(previous_action + xy_)] + max_reward

        return max(reward_action.items(), key=lambda x: x[1])

    def generate(self):
        best_policy = None
        best_reward = -float('inf')
        self.train()
        best_policy = self.search_best_policy()
        best_reward = self.env.evaluatePolicy(best_policy)

        print('=== BEST POLICY ===')
        print(best_policy, best_reward)
        print('===================')

        return best_policy, best_reward


if __name__ == '__main__':
    EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, CustomAgent, "res1.csv")
    EvaluateChallengeSubmission(ChallengeProveEnvironment, CustomAgent, "res2.csv")
