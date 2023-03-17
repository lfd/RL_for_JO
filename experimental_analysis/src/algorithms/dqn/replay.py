"""Memory replay utilities."""

import random
from dataclasses import dataclass, astuple

import tensorflow as tf
import numpy as np

from typing import Any, Iterable, Optional


@dataclass
class Transition:
    state: tf.Tensor
    action: int
    reward: float
    is_terminal: bool
    next_state: Optional[tf.Tensor]
    mask: np.array
    next_mask: np.array


@dataclass
class Batch:
    states: tf.Tensor
    actions: tf.Tensor
    rewards: tf.Tensor
    is_terminal: tf.Tensor
    next_states: tf.Tensor
    masks: tf.Tensor
    next_masks: tf.Tensor


class ReplayMemory:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items = [] 
        self._idx = 0


    def store(self, transition: Transition) -> None:

        item = astuple(transition)

        # Always appending is probably very memory inefficient
        if len(self) < self.capacity:
            self._items.append(item)
        else:
            self._items[self._idx] = item

        self._idx = (self._idx + 1) % self.capacity


    def sample(self, batch_size: int) -> Batch:
        """Sample a batch of transitions, uniformly at random"""

        assert len(self._items) >= batch_size, (
            f'Cannot sample batch of size {batch_size} from buffer with'
            f' {len(self)} elements.'
        )

        # Sample a new batch without replacement
        sampled = random.sample(self._items, batch_size)

        # Transpose List[Transition] -> List[State], List[Action], etc.
        states, actions, rewards, is_terminal, next_states, masks, next_masks = zip(*sampled)

        if isinstance(states[0], list):
            num_inputs = len(states[0])
            s = [[] for _ in range(num_inputs)]
            for i in range(num_inputs):
                [ s[i].append(states[j][i]) for j in range(len(states)) ]
                s[i] = tf.stack(s[i])
            states = s

            s = [[] for _ in range(num_inputs)]
            for i in range(num_inputs):
                for n_state in next_states:
                    if n_state is not None:
                        s[i].append(n_state[i])
                s[i] = tf.stack(s[i])
            next_states = s
        else:
            states = tf.stack(states)
            next_states = tf.stack([s for s in next_states if s is not None])

        masks = tf.stack([s for s in masks if s is not None])
        next_masks = tf.stack([s for s in next_masks if s is not None])

        batch = Batch(
            states = states,
            actions = tf.convert_to_tensor(actions),
            rewards = tf.convert_to_tensor(rewards),
            is_terminal = tf.convert_to_tensor(is_terminal),
            next_states = next_states,
            masks = masks,
            next_masks=next_masks,
        )

        return batch


    def __len__(self) -> int:
        return len(self._items)


    def __iter__(self) -> Iterable[Any]:
        return iter(self._items)


    def __str__(self) -> str:
        return str(self._items)
