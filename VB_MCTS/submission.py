#!/usr/bin/python
# encoding: utf-8

####################################################################################
#  KDD CUP 2019| Policy Learning for Malaria Control
#  Created by Lixin Zou on 2019/7/7,1:40 PM.
#  Copyright © 2019 Lixin Zou (zoulixin15@gmail.com). All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Tsinghua University nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################################################################

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,ExpSineSquared
from sklearn.gaussian_process.kernels import ConstantKernel as C
from netsapi.challenge import *
import warnings
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import copy as cp

RESOLUTION4POLICY = 0.1


def actionSpace(resolution):
    """
    generating the set of actions with specific resolution
    :param resolution: the resolution for solution
    :return: the list of actions
    """
    x, y = np.meshgrid(np.arange(0, 1 + resolution, resolution), np.arange(0, 1 + resolution, resolution))
    xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    return xy.round(2).tolist()

def feature_map(state=1, a_old=[0.0, 0.0], r_old=0.0, act=[1.0, 1.0]):
    """
    the feature map for state
    :param : current state number, last action, last reward, current action
    :return: the feature map
    """
    r1 = [state%2,state%3,state]
    r2 = [r_old]
    r3 = act+a_old
    r4 = [act[0]*(1-a_old[0]),act[1]*(1.-a_old[1])]
    r5 = [act[0]*act[1],a_old[0]*a_old[1],act[1]*a_old[0],act[0]*a_old[1]]
    return np.asarray(r1+r2+r3+r4+r5)

class GpTs(object):
    def __init__(self):
        """
        The KDD CUP 2019 | Policy Learning for Malaria Control
        Learning the policy for Malaria Control with Gaussian Process modeling environment and Tree Search
        """
        self.memory = []
        self.action_space = actionSpace(RESOLUTION4POLICY)
        self.update_count = 0
        self.gprs = []

    def _generate_next_action_feature(self,state,a_old,r_old):
        """
        generating the feature for the next stage
        :param state: current state
        :param a_old: last action
        :param r_old: last reward
        :return: return the action space and corresponding feature
        """
        rf = [feature_map(state=state,a_old=a_old,r_old=r_old,act=item) for item in self.action_space]
        return self.action_space,rf

    def select_initial_action(self):
        """
        generating the initial training samples for Gaussian Process
        :return: return the training samples
        """
        res = [[1,[0.0,0.0],0,item] for item in actionSpace(1.0)]+[[1,[0.0,0.0],0,[0.5,0.5]]]
        return res

    def update_training_sample(self, state, a_old, r_old, act, rwd):
        """
        storing the training samples for gaussian process
        :param current state, last action, last reward, current action, last reward
        :return: None
        """
        f = feature_map(state, a_old, r_old, act)
        self.memory.append([f, rwd])

    def fitting_gaussian_process(self,x_train,y_train):
        """
        Given the training samples, fitting a gaussian process model
        :param x_train: training x
        :param y_train: training y
        :return: the learned gaussian model
        """
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones((feature_map().shape[0],)),
                                                   length_scale_bounds=(0.01, 10.0e20), nu=1.5)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr.fit(x_train,y_train)
        gpr.optimizer = None
        return gpr

    def update_model(self):
        """
          updating gaussian process model with the collected training samples.
          using KFold cross validation to avoid the overfitting of the lengthscale for gaussian process.
          :return: None
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = []
            y = []
            for f,r in self.memory:
                x.append(f)
                y.append(r)
            x = np.asarray(x)
            y = np.asarray(y)
            total_train = x.shape[0]
            if total_train<=20:
                kf = KFold(n_splits=min([x.shape[0],5]))
                for train_index, test_index in kf.split(x):
                    gpr = self.fitting_gaussian_process(x[train_index], y[train_index])
                    self.gprs.append(gpr)
            else:
                kf = KFold(n_splits=min([x.shape[0],5]))
                for train_index, test_index in kf.split(x):
                    gpr = self.fitting_gaussian_process(x[test_index], y[test_index])
                    self.gprs.append(gpr)
            loss = [item.score(x,y) for item in self.gprs]
            best_gpr = self.gprs[np.argmax(loss)]
            kernel = C(best_gpr.kernel_.k1.constant_value, (1e-3, 1e3)) * \
                     Matern(length_scale=best_gpr.kernel_.k2.length_scale,
                            length_scale_bounds=(0.01, 10.0e20), nu=1.5)
            self.gp_reward = GaussianProcessRegressor(kernel=kernel,optimizer=None,n_restarts_optimizer=10)
            self.gp_reward.fit(x,y)
            print(np.sqrt(self.gp_reward.kernel_.k1.constant_value))
            print(self.gp_reward.kernel_.k2.length_scale)

    def select_next_action_search(self,cur_policy,rwds,ratio):
        """
        generating the next action with its estimated reward + ratio*variance,
        top-K is used to select the top-k actions
        :param current_policy: the current branch policy for solution
        :param rwds: the historical rewards
        :param ratio: the ratio for combining mean and variance
        :return: the tree node for expansion
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            next_action, feature = self._generate_next_action_feature(len(cur_policy),cur_policy[-1],rwds[-1])
            q, var = self.gp_reward.predict(feature, return_std=True)
            ucb = q + ratio * var
            tmp = sorted(enumerate(ucb), key=lambda x: x[1], reverse=True)
            res = [(next_action[i], score,q[i]) for i, score in tmp[:50]]
            return res

    def tree_search(self, ratio, step_size = 3 , discount = 0.5,selection = 100):
        """
        :param ratio: the ratio for combining mean and variance
        :param step_size: searching depth for generating policies
        :param discount: the discount factor for RL
        :param selection: the branches with maximum values for tree expansion
        :return: return the policy that best matching the expected values
        """
        current_policy = [[[0.0,0.0]]]
        corr_rewards = [[0.0]]
        corr_mean_rewards = [[0.0]]
        count = -1
        while len(current_policy[0])<step_size+1:
            n_next_policy = []
            n_corr_rewards = []
            n_corr_mean_rewards = []
            count+=1
            for i,item in enumerate(current_policy):
                for next_a in self.select_next_action_search(item,corr_mean_rewards[i],ratio):
                    n_next_policy.append(item+[next_a[0]])
                    n_corr_rewards.append(corr_rewards[i]+[next_a[1]])
                    n_corr_mean_rewards.append(corr_mean_rewards[i]+[next_a[2]])
            tmp = sorted(enumerate([np.sum(item) for item in n_corr_rewards]),key=lambda x:x[1],reverse=True)[:selection]
            current_policy = []
            corr_rewards = []
            corr_mean_rewards = []
            for i,item in tmp:
                current_policy.append(n_next_policy[i])
                corr_rewards.append(n_corr_rewards[i])
                corr_mean_rewards.append(n_corr_mean_rewards[i])
        tmp_reward = []
        for item in corr_rewards:
            v = 0
            for r in item[1:][::-1]:
                v = discount*v + r
            tmp_reward.append(v)
        return current_policy[np.argmax(np.asarray(tmp_reward))]

    def monte_carlo_tree_search(self,ratio=10,step_size=3,discount=1.0,selection = 100):
        """
        :param ratio: the ratio for combining mean and variance
        :param step_size: searching depth for generating policies
        :param discount: the discount factor for RL
        :param selection: the branches with maximum values for tree expansion
        :return: return the policy that best matching the expected values
        """
        current_policy = [[[0.0,0.0]]]
        corr_rewards = [[0.0]]
        while len(current_policy[0])<step_size+1:
            n_next_policy = []
            n_corr_rewards = []
            n_expected_rewards = []
            for i,item in enumerate(current_policy):
                for next_a in self.select_next_action_search(item,corr_rewards[i],ratio):
                    new_policy = item+[next_a[0]]
                    new_reward = corr_rewards[i]+[next_a[1]]
                    max_future_reward = self.rollout(len(new_policy),next_a[0],next_a[2],ratio,step_size,100)
                    n_expected_rewards.append(max_future_reward)
                    n_next_policy.append(new_policy)
                    n_corr_rewards.append(new_reward)
            tmp = sorted(enumerate(n_expected_rewards),key=lambda x:x[1],reverse=True)[:selection]
            current_policy = []
            corr_rewards = []
            for i,item in tmp:
                current_policy.append(n_next_policy[i])
                corr_rewards.append(n_corr_rewards[i])
        tmp_reward = []
        for item in corr_rewards:
            v = 0
            for r in item[1:][::-1]:
                v = discount*v + r
            tmp_reward.append(v)
        return current_policy[np.argmax(np.asarray(tmp_reward))]

    def rollout(self,state,a_old,r_old,ratio,step_size,rollout_num=1000):
        """
        rollout at a specific root node to evalute the goodness of a given node
        :param state: current state
        :param a_old: last action
        :param r_old: last reward
        :param ratio: the ratio for trading off between mean and variance
        :param step_size: the step size for rollout
        :param rollout_num: the number of rollout
        :return: return the maximum future reward for a given node
        """
        if state > step_size: return 0+r_old
        trwd = []
        s = state
        a_ot = [a_old]*rollout_num
        r_ot = [r_old]*rollout_num
        while s<=5:
            acts = [self.action_space[key] for key in np.random.choice(range(len(self.action_space)),(rollout_num,))]
            rwd,r_v = self.predict_batch_reward_variance([s]*rollout_num,a_ot,r_ot,acts)
            trwd.append(rwd+np.sqrt(r_v)*ratio)
            a_ot = acts
            r_ot = rwd
            s+=1
        trwd = np.sum(trwd,axis=0)
        return np.max(trwd)+r_old

    def predict_reward_variance(self,state,a_old,r_old,act):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = feature_map(state,a_old,r_old,act)
            return self.gp_reward.predict([rf],return_std=True)

    def predict_batch_reward_variance(self,state,a_old,r_old,act):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = [feature_map(state[i],a_old[i],r_old[i],item) for i,item in enumerate(act)]
            return self.gp_reward.predict(rf,return_std=True)

class GpTsAgent(object):
    def __init__(self, environment):
        self.env = environment
        self.model = GpTs()

    def train(self):
        """
        finding the policy for Malaria Control
        :return:
        """
        self.count = 0
        for state,act1,r1,act2 in self.model.select_initial_action():
            self.env.reset()
            _,r2,_,_ = self.env.evaluateAction(act2)
            self.count+=1
            self.model.update_training_sample(state,act1,r1,act2,r2)
        self.model.update_model()
        self.policies = []
        self.rewards = []
        ratio = 3.5
        step_size = 1.0
        while self.count<=104:
            print("#"*5 + "sampling policy")
            policy = self.model.tree_search(ratio=ratio,step_size = step_size , discount = 1.0, selection = 1000)
            step_size = step_size+1
            if step_size >= 5: step_size = 5
            self.policy_update(policy)
            if step_size>=5:
                print("#" * 5 + "policy evaluation")
                evaluated_policy = self.model.tree_search(ratio=0,step_size=int(step_size),discount=1.0,selection= 1000)
                self.policy_update(evaluated_policy)
            print("#"*5+"finish an epoch")
        return max(self.rewards),self.policies[np.argmax(self.rewards)]

    def policy_update(self,evaluated_policy):
        """
        updating the learned model and evaluated the policy
        :param evaluated_policy
        :return:
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_rwd = [0.0]
            self.env.reset()
            print(evaluated_policy)
            for i, item in enumerate(evaluated_policy[1:]):
                if self.count==105: continue
                s = self.env.state
                mean,variance = self.model.predict_reward_variance(s,evaluated_policy[i],s_rwd[-1],item)
                _, rwd, _, _ = self.env.evaluateAction(item)
                self.count+=1
                print(s_rwd[-1],evaluated_policy[i],evaluated_policy[i + 1],rwd,mean,variance)
                self.model.update_training_sample(s,evaluated_policy[i],s_rwd[-1],item,rwd)
                s_rwd.append(rwd)
            self.model.update_model()
            self.policies.append({str(i + 1): evaluated_policy[1:][i] for i in range(len(evaluated_policy[1:]))})
            self.rewards.append(np.sum(s_rwd))
            print(evaluated_policy, self.rewards[-1])

    def generate(self):
        """
        generating the policy and expected reward
        :return:
        """
        best_reward,best_policy = self.train()
        print(best_policy, best_reward)
        return best_policy,best_reward

if __name__ == '__main__':
    # for _ in range(10):
    #     env = ChallengeSeqDecEnvironment(1000)
    #     GpTsAgent(env)
    EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, GpTsAgent, "res1.csv")
    EvaluateChallengeSubmission(ChallengeProveEnvironment, GpTsAgent, "res2.csv")
