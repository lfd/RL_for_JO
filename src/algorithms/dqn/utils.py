import numpy as np
from math import log
import random
import tensorflow as tf

from src.algorithms import utils

from .policies import Policy
from .replay import Transition

class Sampler:

    def __init__(self, policy: Policy, env, use_mask: bool = True, take_best_threshold = 0) -> None:
        self._policy = policy
        self._env = env

        self._state, self.info = self._env.reset()
        self.use_mask = use_mask
        self.take_best_threshold = take_best_threshold
        self.take_best = random.random() < take_best_threshold
        self.idx = 0


    def step(self) -> Transition:
        """Samples the next transition"""

        mask = self._env.get_mask() if self.use_mask else None
        state_info = to_tuple(self._state)

        curr_states = self._state if self.use_mask else self._env.get_all_next_states()
        if self.take_best:
            best_action = self._env.get_best_action(self.idx)
            action = tf.convert_to_tensor([best_action], dtype=tf.int64)
        else:
            action, _ = self._policy(curr_states, mask)

        next_state, reward, t1, t2, _ = self._env.step(action[0].numpy())
        done = t1 or t2

        transition = Transition(
            state=self._state,
            reward=reward,
            is_terminal=done,
            action=action[0],
            next_state=curr_states if not done else None,
            mask=mask,
            next_mask=self._env.get_mask() if not done else None,
        )

        if done:
            self._state, self.info = self._env.reset()
            self.idx = 0
            self.take_best = random.random() < self.take_best_threshold
        else:
            self._state = next_state
            self.idx += 1

        if self.info["reached_end"]:
            return None

        return transition

    def update_env(self, env):
        self._env = env
        self._state, self._info = self._env.reset()

def to_tuple(state):
    flattened_state = np.concatenate(state)
    tuple_state = tuple(flattened_state)
    return(tuple_state)
