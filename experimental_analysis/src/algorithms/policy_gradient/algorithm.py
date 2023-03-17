import tensorflow as tf
import numpy as np
from collections import defaultdict
from copy import deepcopy, copy

from src.envs.join_order.environment import JoinOrdering
from src.configuration import Configuration
from src.logger import Logger
from src.algorithms import utils

def train(conf: Configuration, weight_file: str = None):

    """
    Policy gradient reinforcement learning algorithm -REINFORCE-
    """
    logger = Logger(conf.experiment_name)

    policy_model = conf.policy_model

    if weight_file:
        policy_model.set_weights(utils.get_model_weights_from_file(weight_file))

    optimizers, trainable_variables = conf.get_optimizers()

    reward_history = []

    num_inputs = len(policy_model.inputs)
    iteration = 0

    env = conf.env
    val_env = conf.val_env

    for batch in range(conf.num_updates):
        buffer, sum_return, sum_length, iteration = utils.gather_trajectories(policy_model, env, conf, iteration=iteration)

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
            elems_in_buffer,
        ) = buffer.get()

        logger.log_batch(batch, conf.num_episodes, sum_return / conf.num_episodes, sum_rewards=sum_return)

        # calculate gradient
        with tf.GradientTape() as tape:
            tape.watch(policy_model.trainable_variables)
            logits = policy_model(observation_buffer)
            log_probs = utils.logprobabilities(logits, action_buffer, conf.num_actions)
            loss = tf.reduce_mean(-log_probs * return_buffer)

        grads = tape.gradient(loss, policy_model.trainable_variables)

        # Update model parameters.
        for idx, optimizer in enumerate(optimizers):
            optimizer.apply_gradients(
                    [(grads[idx], trainable_variables[idx])]
            )

        if batch % conf.save_every == 0:
            logger.save_model_parameters(policy_model.get_weights(), batch)

        if isinstance(val_env, JoinOrdering) and batch % conf.validate_every == 0:
            utils.validate_policy(policy_model, val_env, batch, logger, conf)

        if iteration >= conf.total_timesteps:
            break

    if isinstance(val_env, JoinOrdering):
        utils.validate_policy(policy_model, val_env, batch, logger, conf)
    logger.save_model_parameters(policy_model.get_weights(), batch)

