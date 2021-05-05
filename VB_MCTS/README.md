# XXXX_2019
## Our Solution:
    
Our solution is Gaussian Process + MCTS. First, using Gaussian Process to predict the reward function. Then, chosing the action by MCTS for maximizing the mean reward + variance of the learned reward function (approximate the UCB in finite horizon MDP). Tree search with pruning can be used to replace MCTS for fastering the training process. 

## Check phase
Our method can find the policy [[0.0,0.8],[1.0,0.0],[0.0,0.8],[1.0,0.0],[0.0,0.8]]. The reward is around 550.

## Verification phase
Our method can find the policy [[0.2,0.9],[xx,xx],[xx,xx],[xx,xx],[xx,xx]]. The reward is around 300.
Sometimes, we can find a better policy [[0.2,0.9],[xx,xx],[xx,xx],[xx,xx],[0.0,0.5]], the reward is around 550. However, this law of getting a bigger reward in the 5-step is hard to generalize to other states. 
