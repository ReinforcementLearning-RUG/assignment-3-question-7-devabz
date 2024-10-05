import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.abstract_evaluator import AbstractEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


class TDLambdaEvaluator(AbstractEvaluator):
    def __init__(self,
                 env: AbstractMDP,
                 alpha: float,
                 lambd: float):
        """
        Initializes the TD(λ) Evaluator.

        :param env: A mdp object.
        :param alpha: The step size.
        :param lambd: The trace decay parameter (λ).
        """
        self.env = env
        self.alpha = alpha
        self.lambd = lambd
        self.value_fun = np.zeros(self.env.num_states)    # Estimate of state-value function.
        self.eligibility_traces = np.zeros(self.env.num_states)

    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> np.ndarray:
        """
        Perform the TD prediction algorithm.

        :param policy: A policy object that provides action probabilities for each state.
        :param num_episodes: Number of episodes to run for estimating V(s).
        :return: The state-value function V(s) for the associated policy.
        """
        self.value_fun.fill(0)              # Reset value function.

        ep_rewards = []
        ep_length = []
        for _ in range(num_episodes):
            ep_reward, steps = self._update_value_function(policy)
            ep_rewards.append(ep_reward)
            ep_length.append(steps)
            
        return self.value_fun.copy(), ep_rewards, ep_length

    def _update_value_function(self, policy: AbstractPolicy) -> None:
        """
        Runs a single episode using the TD(λ) method to update the value function.

        :param policy: A policy object that provides action probabilities for each state.
        """
        gamma = self.env.discount_factor
        state, t = self.env.reset(), 0
        terminal = False
        ep_reward = 0
        steps = 0
        self.eligibility_traces = np.zeros_like(self.eligibility_traces)
        while not terminal:
            t += 1
            steps += 1
            action = policy.sample_action(state=state)
            next_state, reward, terminal = self.env.step(action)
            td_target = reward + gamma * self.value_fun[next_state]
            td_error = td_target - self.value_fun[state]
            self.value_fun[state] = self.value_fun[state] + self.alpha * self.eligibility_traces[state] * td_error
            state = next_state
            ep_reward += reward
            self.eligibility_traces[state] += 1
            self.eligibility_traces *= self.lambd
            
        return ep_reward, steps
        
