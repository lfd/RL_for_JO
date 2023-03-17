import tensorflow as tf
import numpy as np
from itertools import count

from .policies import ActionValuePolicy, EpsilonGreedyPolicy, LinearDecay, ActionValueNextStatesPolicy
from .replay import ReplayMemory
from .utils import Sampler
from src.logger import Logger
from .validate import validate
from src.configuration import Configuration
from src.algorithms import utils
from src.envs.join_order.environment import JoinOrdering

def train(conf: Configuration, weight_file: str = None):
        """
        The Deep Q-Learning algorithm.

        Args:
            conf: The training configuration
        """
        logger = Logger(conf.experiment_name)

        env = conf.env
        val_env = conf.val_env

        policy_model = conf.policy_model
        target_model = conf.target_model

        if weight_file:
            policy_model.set_weights(utils.get_model_weights_from_file(weight_file))
            target_model.set_weights(utils.get_model_weights_from_file(weight_file))

        greedy_policy = ActionValuePolicy(policy_model) if conf.mask else ActionValueNextStatesPolicy(policy_model)
        behavior_policy = EpsilonGreedyPolicy(greedy_policy, conf.num_actions)

        memory = ReplayMemory(conf.replay_capacity)

        epsilon_schedule = LinearDecay(
            policy = behavior_policy,
            num_steps = conf.epsilon_duration,
            start = conf.epsilon_start,
            end = conf.epsilon_end
        )

        optimizers, trainable_variables = conf.get_optimizers()

        # for optimizer in optimizers:
        #     optimizer.build(trainable_variables)

        if conf.prefill_buffer:
            prefill_sampler = Sampler(behavior_policy, conf.incr_train_env, use_mask=conf.mask, take_best_threshold=1)
            transition = prefill_sampler.step()
            while transition is not None:
                memory.store(transition)
                transition = prefill_sampler.step()

        sampler = Sampler(behavior_policy, env, use_mask=conf.mask, take_best_threshold=conf.take_best_threshold)
        episode = 0
        transitions = []

        # Training Loop
        for step in count():

            is_training = episode - conf.train_after >= 0

            if is_training:
                epsilon_schedule.step()

            transition = sampler.step()
            # if conf.prio and transition.reward < 0:
            #     for _ in range(-int(transition.reward) + 1):
            #         memory.store(transition)
            # else:

            if True: # if transition.reward == 0:
                memory.store(transition)
                # transitions.append(transition)
            else:
                for t in transitions:
                    t.reward = transition.reward
                    memory.store(t)
                transitions = []
                memory.store(transition)

            logger.log_transition(
                transition=transition,
                did_explore=behavior_policy.did_explore
            )

            if transition.is_terminal:
                episode += 1
                reached_terminal_update = True
                reached_terminal_val = True

            if not is_training:
                continue

            if step % conf.train_every == 0:

                # Sample a batch of transitions from the replay buffer
                batch = memory.sample(conf.batch_size)

                # Check whether the batch contains next_states (the sampled
                # batch might contain terminal states only)
                if not isinstance(batch.next_states, list) and len(batch.next_states) > 0 or \
                        len(batch.next_states[0]) > 0:

                    target_next_q_values = utils.sample_values(target_model, batch.next_states, batch.next_masks)

                    target_next_v_values = tf.reduce_max(
                        target_next_q_values,
                        axis=-1
                    )

                    non_terminal_indices = tf.where(~batch.is_terminal)

                    targets = tf.tensor_scatter_nd_add(
                        batch.rewards,
                        non_terminal_indices,
                        conf.gamma * target_next_v_values
                    )
                else:
                    targets = batch.rewards

                # targets = batch.best_values

                with tf.GradientTape() as tape:
                    if conf.mask:
                        policy_q_values = utils.sample_values(policy_model, batch.states)

                        action_indices = tf.expand_dims(batch.actions, axis=-1)

                        policy_v_values = tf.gather(
                            policy_q_values,
                            action_indices,
                            batch_dims=1
                        )

                    else:
                        # policy_q_values = utils.sample_values(policy_model, batch.next_states)

                        # sliced_q_values = []
                        # ptr = 0
                        # for i, num_a in enumerate(batch.num_actions):
                        #     sliced_q_values.append(tf.gather(policy_q_values[ptr : ptr+num_a], batch.actions[i]))
                        #     ptr += num_a

                        # policy_v_values = tf.stack(sliced_q_values, axis=1)[0]

                        policy_v_values = utils.sample_values(policy_model, batch.states)

                    policy_v_values = tf.squeeze(
                        policy_v_values,
                        axis=-1
                    )

                    loss = conf.loss(targets, policy_v_values)

                grads = tape.gradient(
                    loss,
                    trainable_variables
                )

                for idx, optimizer in enumerate(optimizers):
                    optimizer.apply_gradients(
                            [(grads[idx], trainable_variables[idx])]
                    )

                logger.log_training(
                    step=step,
                    loss=np.mean(loss.numpy()),
                    batch=batch
                )

            if step % conf.update_every == 0:

                target_model.set_weights(
                    policy_model.get_weights()
                )

            if episode % conf.validate_every == 0 and reached_terminal_val:
                reached_terminal_val = False

                if isinstance(val_env, JoinOrdering):
                    utils.validate_policy(policy_model, val_env, episode, logger, conf)
                    utils.validate_policy(policy_model, conf.incr_train_env, episode, logger, conf, training=True)

                else:
                    val_return = validate(
                        val_env,
                        greedy_policy,
                        conf.num_val_steps,
                        conf.num_val_trials,
                        logger
                    )

                    logger.log_validation(
                        val_return=val_return,
                        grads=grads,
                        trainable_variables=trainable_variables
                    )

            if episode % conf.update_dataset_every == 0 and reached_terminal_update:
                reached_terminal_update = False
                utils.update_training_set(env, policy_model, conf)
                sampler.update_env(env)

            if episode % conf.save_every == 0:
                logger.save_model_parameters(policy_model.get_weights(), episode)

            if episode >= conf.num_episodes:
                break

