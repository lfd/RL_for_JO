"""Classes to represent Reinforcement Learning policies."""

import abc
import random

import tensorflow as tf
import numpy as np

from typing import Union, List

from src.algorithms import utils

class Policy(abc.ABC):
    """Abstract base class for policies."""

    @abc.abstractmethod
    def action(self, state: tf.Tensor):
        """Returns action if len(states) == 1"""
        pass

    def __call__(self, state: tf.Tensor, mask=None):
        return self.action(state, mask)


class ActionValuePolicy(Policy):
    """Selects the greedy action with respect to a set of action values (a.k.a. 
    Q-values), predicted by a model based on a given state."""

    def __init__(self, model: tf.Module):
        super(ActionValuePolicy, self).__init__()
        self.model = model


    def values(self, states: tf.Tensor, mask) -> tf.Tensor:
        return utils.sample_values(self.model, states, mask)

    def action(self, state: tf.Tensor, mask=None):
        inputs = [tf.expand_dims(s, axis = 0) for s in state] if isinstance(state, list) else tf.expand_dims(state, axis=0) 

        q_values = self.values(inputs, mask)

        action = tf.argmax(q_values, axis=1)

        q_values=q_values[0]
        return action, q_values

class ActionValueNextStatesPolicy(Policy):
    """Same as ActionValuePolicy, but assumes that the policy model only
    predicts one Q-value for the current observations.
    Q-values for all action are predicted for each possible next state."""

    def __init__(self, model: tf.Module):
        super(ActionValueNextStatesPolicy, self).__init__()
        self.model = model

    def values(self, all_next_states) -> tf.Tensor:
        return utils.sample_values(self.model, all_next_states)

    def action(self, all_next_states, mask=None):
        if isinstance(all_next_states, list):

            if isinstance(all_next_states[0], list):
                l = [[] for _ in range(len(all_next_states[0]))]
                for i in range(len(all_next_states[0])):
                    [l[i].append(s[i]) for s in all_next_states]
                inputs = [tf.convert_to_tensor(s) for s in l]
            else:
                inputs = [tf.expand_dims(s, axis = 0) for s in all_next_states]

        else:
            tf.expand_dims(state, axis=0)

        q_values = self.values(inputs)

        action = tf.argmax(q_values, axis=0, output_type=tf.int32)

        q_values=q_values[0]
        return action, q_values


class EpsilonGreedyPolicy(Policy):
    """Selects an action from the underlying `policy` with a probability of
    `epsilon`, or uniformly at random otherwise."""

    def __init__(self, policy: Policy, num_actions: int, epsilon: float = 1.0):
        super(EpsilonGreedyPolicy, self).__init__()

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f'Epsilon must be in range[0,1], but was {epsilon}')

        self.policy = policy
        self.num_actions = num_actions
        self.epsilon = epsilon

        self._did_explore = False


    @property
    def did_explore(self):
        return self._did_explore


    def action(self, state: tf.Tensor, mask=None):
        self._did_explore = random.random() < self.epsilon
        if mask is None:
            action = tf.convert_to_tensor([np.random.randint(len(state))]) if self.did_explore else self.policy(state, mask)
        else:
            action = utils.sample_random_action(self.num_actions, mask) if self.did_explore else self.policy(state, mask)

        if isinstance(action, tuple):
            return action
        else:
            return action, None


class LinearDecay:
    """Let's the `epsilon` parameter of the underlying `EpsilonGreedyPolicy`
    decay linearly from a `start` to `end` over a set number of steps."""

    def __init__(self, policy: EpsilonGreedyPolicy, num_steps: int, start: float = 1.0, end: float = 0.0):

        if not 0.0 <= start <= 1.0:
            raise ValueError(f'start must be in range [0,1], but was {start}')

        if not 0.0 <= end <= 1.0:
            raise ValueError(f'end must be in range [0,1], but was {end}')

        if num_steps <= 0:
            raise ValueError(f'num_steps must be a positive integer, but was {num_steps}')

        self.policy = policy
        self.start = start 
        self.end = end
        self.num_steps = num_steps

        self.policy.epsilon = self.start

        self._step = 0
        self._slope = (end - start) / (num_steps - 1)


    def step(self) -> None:

        self._step += 1

        if self._step < self.num_steps:
            self.policy.epsilon = self.start + self._slope * self._step

