"""
Memory buffer utility
Similar to standard Q-learning, information of transitions are stored within
the buffer. However, not a full transition is stored, but only the state and an
identifier to retrieve the best reward achived in this state so far.
"""
from dataclasses import dataclass, astuple
import numpy as np
import tensorflow as tf
from typing import Iterable, Any, List
import random

@dataclass
class Transition:
    state: tf.Tensor
    reward: float
    state_info: str
    is_terminal: bool
    action: int
    next_states: list
    num_actions: int

@dataclass
class Batch:
    states: tf.Tensor
    best_values: tf.Tensor
    actions: tf.Tensor
    next_states: tf.Tensor
    num_actions: List[int]

class Buffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items = []
        self._idx = 0
        self.best_rewards = dict()

    def store(self, transition: Transition) -> None:

        if transition.state_info not in self.best_rewards or \
                self.best_rewards[transition.state_info] > transition.reward:
            self.best_rewards[transition.state_info] = transition.reward
        item = astuple(transition)

        # Always appending is probably very memory inefficient
        if len(self) < self.capacity:
            self._items.append(item)
        else:
            self._items[self._idx] = item

        self._idx = (self._idx + 1) % self.capacity

    def sample(self, batch_size: int) -> Batch:
        """Sample a batch from buffer, uniformly at random"""

        assert len(self._items) >= batch_size, (
            f'Cannot sample batch of size {batch_size} from buffer with'
            f' {len(self)} elements.'
        )

        # Sample a new batch without replacement
        sampled = random.sample(self._items, batch_size)

        # Transpose List[Transition] -> List[State], List[Values], etc.
        states, values, value_infos, is_terminal, actions, all_next_states, num_actions = zip(*sampled)

        if isinstance(states[0], list):
            num_inputs = len(states[0])
            s = [[] for _ in range(num_inputs)]
            for i in range(num_inputs):
                [ s[i].append(states[j][i]) for j in range(len(states)) ]
                s[i] = tf.stack(s[i])
            states = s
        else:
            states = tf.stack(states)

        if isinstance(all_next_states[0][0], list):
            num_inputs = len(all_next_states[0][0])
            s = [[] for _ in range(num_inputs)]
            for i in range(num_inputs):
                for j in range(batch_size):
                    [ s[i].append(all_next_states[j][k][i]) for k in range(num_actions[j]) ]
                s[i] = tf.stack(s[i])
            all_next_states = s
        else:
            all_next_states = None #tf.stack(all_next_states)

        best_values = np.zeros(batch_size)
        for i in range(batch_size):
            best_values[i] = self.best_rewards[value_infos[i]]

        best_values = tf.convert_to_tensor(best_values)

        batch = Batch(
            states = states,
            best_values = best_values,
            actions = tf.convert_to_tensor(actions),
            next_states=all_next_states,
            num_actions=num_actions,
        )

        return batch


    def __len__(self) -> int:
        return len(self._items)


    def __iter__(self) -> Iterable[Any]:
        return iter(self._items)


    def __str__(self) -> str:
        return str(self._items)

