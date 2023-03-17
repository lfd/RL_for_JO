import tensorflow as tf
import numpy as np
from collections import defaultdict
from copy import deepcopy, copy
import tensorflow_probability as tfp

from src.envs.join_order.environment import JoinOrdering
from src.configuration import Configuration
from src.logger import Logger
from src.algorithms import utils
from src.algorithms.policy_gradient.buffer import Buffer

def train(conf: Configuration, weight_file: str = None):

    """
    Proximal Policy Optimisation Algorithm
    https://keras.io/examples/rl/ppo_cartpole
    """
    logger = Logger(conf.experiment_name)

    env = conf.env
    val_env = conf.val_env
    policy_model = conf.policy_model
    critic_model = conf.critic_model

    if weight_file:
        policy_model.set_weights(utils.get_model_weights_from_file(weight_file))

    optimizers, trainable_variables = conf.get_optimizers()

    num_inputs = conf.num_inputs

    for batch in range(conf.num_updates):

        buffer, sum_return, sum_length, costs, DP_costs, ex_times, pg_ex_times, geqo_costs = \
                utils.gather_trajectories(policy_model, env, conf, critic_model, iteration=batch)

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
            elems_in_buffer,
            mask_buffer,
        ) = buffer.get()

        # Index of each element of batch_size
        # Create the indices array
        inds = np.arange(elems_in_buffer)

        logger.log_batch(batch, conf.num_episodes, sum_return / conf.num_episodes, sum_return, costs, DP_costs, ex_times, pg_ex_times, geqo_costs)

        # Update the policy
        for i in range(conf.train_iterations):
            num_minibatches = elems_in_buffer // conf.mini_batchsize + 1
            num_minibatches -= 1 if elems_in_buffer % conf.mini_batchsize == 0 else 0

            policy_losses, value_losses, entropies, losses = [np.zeros(num_minibatches) for _ in range(4)]
            np.random.shuffle(inds)

            # 0 to batch_size with batch_train_size step
            for j, start in enumerate(range(0, elems_in_buffer, conf.mini_batchsize)):
                end = start + conf.mini_batchsize
                if end >= elems_in_buffer:
                    end = elems_in_buffer
                mbinds = inds[start:end]

                if isinstance(observation_buffer, list):
                    ob_mbinds = [o[mbinds] for o in observation_buffer]
                else:
                    ob_mbinds = observation_buffer[mbinds]

                grads, policy_loss, value_loss, entropy, loss = train_models(
                    policy_model,
                    critic_model,
                    optimizers, trainable_variables,
                    conf.v_coeff, conf.ent_coeff,
                    ob_mbinds, action_buffer[mbinds],
                    logprobability_buffer[mbinds], advantage_buffer[mbinds],
                    return_buffer[mbinds],
                    conf.num_actions, conf.clip_ratio,
                    mask_buffer[mbinds]
                )
                policy_losses[j] = policy_loss
                value_losses[j] = value_loss
                entropies[j] = entropy
                losses[j] = loss

            logger.log_minibatch_ppo(batch,
                                     batch*conf.train_iterations + i,
                                     sum_return / conf.num_episodes,
                                     policy_loss=np.mean(policy_losses), entropy=np.mean(entropies),
                                     loss=np.mean(losses), grads=grads,
                                     policy_trainable_variables=policy_model.trainable_variables,
                                     value_loss=np.mean(value_losses),
                                     critic_trainable_variables=critic_model.trainable_variables)

        if batch % conf.save_every == 0:
            logger.save_model_parameters(policy_model.get_weights(), batch, critic_model.get_weights())

        if isinstance(val_env, JoinOrdering) and batch % conf.validate_every == 0:
            utils.validate_policy(policy_model, val_env, batch, logger, conf)
            # utils.validate_policy(policy_model, conf.incr_train_env, batch, logger, conf, training=True)

        if (batch*conf.num_episodes) % conf.update_dataset_every == 0:
            utils.update_training_set(env, policy_model, conf)

    if isinstance(val_env, JoinOrdering):
        utils.validate_policy(policy_model, val_env, batch, logger, conf)
    logger.save_model_parameters(policy_model.get_weights(), batch, critic_model.get_weights())


# Train the policy by maxizing the PPO-Clip objective
# and train critic model by minimising MSE
def train_models(
    policy_model, value_model,
    optimizers, trainable_variables,
    v_coeff, ent_coeff,
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer,
    return_buffer,
    num_actions, clip_ratio,
    mask_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        logits, _ = utils.sample_action(policy_model, observation_buffer, mask=mask_buffer)

        probs = tfp.distributions.Categorical(logits=logits)
        entropy = tf.reduce_mean(probs.entropy())

        log_probs = utils.logprobabilities(logits, action_buffer, num_actions)
        ratio = tf.exp(log_probs - logprobability_buffer)

        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
        value_loss = tf.reduce_mean((return_buffer - value_model(observation_buffer)) ** 2)

        loss = policy_loss + v_coeff * value_loss - ent_coeff * entropy

    grads = tape.gradient(loss, trainable_variables)

    for idx, optimizer in enumerate(optimizers):
        optimizer.apply_gradients(
                [(grads[idx], trainable_variables[idx])]
        )

    return grads, policy_loss, value_loss, entropy, loss


