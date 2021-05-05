XXXX 2019 | Policy Learning for Malaria Control
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