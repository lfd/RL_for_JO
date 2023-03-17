import pickle
import tensorflow as tf
from random import random

from src.algorithms.policy_gradient.buffer import Buffer
from src.envs.join_order.environment import JoinOrdering
from src.configuration import DQNConfiguration

def get_model_weights_from_file(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Sample action
def sample_action(policy_model, observation, mask = None):
    logits = policy_model(observation)
    if mask is not None:
        mask = tf.cast(mask, tf.bool)
        neg_inf = tf.fill(dims=tf.shape(logits), value=logits.dtype.min)
        logits = tf.where(mask, logits, neg_inf)
    logits = tf.nn.softmax(logits)
    if tf.math.reduce_any(tf.math.is_nan(logits)):
        ones = tf.fill(dims=tf.shape(logits), value=1.)
        if mask is not None:
            logits = tf.where(mask, ones, neg_inf)
        else:
            logits = ones
        logits = tf.nn.softmax(logits)
    logits, norm = tf.linalg.normalize(logits, ord=1)
    action = tf.squeeze(tf.random.categorical(tf.math.log(logits), 1), axis=1)
    return logits, action

@tf.function
def sample_values(policy_model, observation, mask = None):
    vals = policy_model(observation)
    if mask is not None and mask.shape != 0:
        mask = tf.cast(mask, tf.bool)
        neg_inf = tf.fill(dims = tf.shape(vals), value=vals.dtype.min)
        vals = tf.where(mask, vals, neg_inf)
    return vals

@tf.function
def sample_random_action(num_actions, mask = None):
    logits = tf.ones((1,num_actions))
    if mask is not None:
        mask = tf.cast(mask, tf.bool)
        neg_inf = tf.fill(dims=tf.shape(logits), value=logits.dtype.min)
        logits = tf.where(mask, logits, neg_inf)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=0)
    return action

def logprobabilities(logits, a, num_actions):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    one_hot_logit = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logits, axis=1
    )
    logprobability = tf.math.log(one_hot_logit)
    return logprobability

def gather_trajectories(policy_model, env, conf, critic_model = None, iteration = 0, pre_training = False):

    num_episodes = len(env.queries)*2 if pre_training else conf.num_episodes
    buffer = Buffer(conf.num_inputs, max_size=num_episodes*conf.max_num_steps_per_episode, gamma=conf.gamma, action_dimension=conf.num_actions)

    # Initialize the sum of the returns and lengths
    sum_return = 0
    sum_length = 0
    costs = []
    DP_costs = []
    geqo_costs = []
    ex_times = []
    pg_ex_times = []

    for episode in range(num_episodes):

        # Initialize the observation, episode return and episode length
        observation, res_info = env.reset()
        reached_end = res_info["reached_end"]
        if reached_end:
            break
        episode_return, episode_length = 0, 0
        done = False
        info = dict()
        idx = 0
        use_best =  (iteration < conf.take_best_warmup or (iteration*conf.num_episodes) % conf.take_best_frequency == 0) \
                and random() < conf.take_best_threshold

        while not done:
            # Get the logits, action, and take one step in the environment
            if isinstance(observation, list):
                observation = [ o.reshape(1, -1) for o in observation]
            else:
                observation = observation.reshape(1,-1)

            mask = env.get_mask() if conf.mask else None

            logits, action = sample_action(policy_model, observation, mask)

            if use_best:
                best_action = env.get_best_action(idx)

                # check if best action is likely to be sampled
                if logits[0][best_action] > 0:
                    best_action = tf.convert_to_tensor([best_action], dtype=action.dtype)
                    action = best_action
                else:
                    use_best = False

            observation_new, reward, t1, t2, info = env.step(action[0].numpy())
            done = t1 or t2
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            if critic_model is None:
                value_t = None
                logprobability_t = None
            else:
                value_t = critic_model(observation)
                logprobability_t = logprobabilities(logits, action, conf.num_actions)

            # Store obs, act, rew, v_t, logp_pi_t, mask
            buffer.store(observation, action, reward, value_t, logprobability_t, mask)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            if isinstance(observation, list):
                obs = [o.reshape(1, -1) for o in observation]
            else:
                obs = observation.reshape(1, -1)

            idx += 1

        cost, ex_time = info.get("cost", -1), info.get("execution_time", -1)
        DP_cost, pg_ex_time = info.get("DP_cost", -1), info.get("pg_execution_time", -1)
        geqo_cost = info.get("geqo_cost", -1)
        costs.append(cost)
        DP_costs.append(DP_cost)
        geqo_costs.append(geqo_cost)
        ex_times.append(ex_time)
        pg_ex_times.append(pg_ex_time)

        print(f"Episode return: {episode_return}")

        buffer.finish_trajectory()
        sum_return += episode_return
        sum_length += episode_length

    return buffer, sum_return, sum_length, costs, DP_costs, ex_times, pg_ex_times, geqo_costs

def validate_policy(policy_model, env, batch, logger, conf, training = False, execution = False):
    state, info = env.reset()
    query_name = info["query_file"]
    reached_end = info["reached_end"]
    rewards = []
    costs = []
    DP_costs = []
    geqo_costs = []
    ex_times = []
    pg_ex_times = []
    geqo_ex_times = []
    num_episodes = 1 if isinstance(conf, DQNConfiguration) else conf.num_episodes

    while not reached_end:
        reward = 0
        done = False

        while not done:
            state = state if conf.mask else env.get_all_next_states()
            if isinstance(state[0], list):
                l = [[] for _ in range(len(state[0]))]
                for i in range(len(state[0])):
                    [l[i].append(s[i]) for s in state]
                curr_state = [tf.convert_to_tensor(s) for s in l]
            else:
                curr_state = [tf.expand_dims(s, axis = 0) for s in state]

            mask = env.get_mask() if conf.mask else None
            policy = policy_model(curr_state).numpy()
            num_actions = policy.shape[1]

            if isinstance(conf, DQNConfiguration):
                vals = sample_values(policy_model, curr_state, mask)
                action = tf.argmax(vals, axis=1 if conf.mask else 0, output_type=tf.int32)
            else:
                _, action = sample_action(policy_model, curr_state, mask)
            state, r, t1, t2, step_info = env.step(action[0])
            done = t1 or t2
            reward += r

        query_name = info["query_file"]
        cost = step_info.get("cost", -1)
        ex_time = step_info.get("execution_time", -1)
        DP_cost = step_info.get("DP_cost", -1)
        geqo_cost = step_info.get("geqo_cost", -1)
        geqo_ex_time = step_info.get("geqo_execution_time", -1)
        pg_ex_time = step_info.get("pg_execution_time", -1)
        if not training or execution:
            logger.validate_policy_on_query(batch, num_episodes, query_name, reward, cost, DP_cost, ex_time, pg_ex_time, geqo_cost, geqo_ex_time)

        rewards.append(reward)
        costs.append(cost)
        DP_costs.append(DP_cost)
        geqo_costs.append(geqo_cost)
        ex_times.append(ex_time)
        pg_ex_times.append(pg_ex_time)

        state, info = env.reset()
        reached_end = info["reached_end"]

    if not execution:
        logger.log_validation_avg(batch, num_episodes, sum(rewards) / len(rewards), costs, DP_costs, ex_times, pg_ex_times, geqo_costs, training)

def update_training_set(env, policy_model, conf):
    env.set_queries_incremental()
    query_set = []
    state, info = env.reset()
    reached_end = info["reached_end"]

    while not reached_end:
        reward = 0
        done = False
        query_set.append(env.query)

        while not done:
            state = state if conf.mask else env.get_all_next_states()
            if isinstance(state[0], list):
                l = [[] for _ in range(len(state[0]))]
                for i in range(len(state[0])):
                    [l[i].append(s[i]) for s in state]
                curr_state = [tf.convert_to_tensor(s) for s in l]
            else:
                curr_state = [tf.expand_dims(s, axis = 0) for s in state]

            mask = env.get_mask() if conf.mask else None
            policy = policy_model(curr_state).numpy()
            num_actions = policy.shape[1]

            if isinstance(conf, DQNConfiguration):
                vals = sample_values(policy_model, curr_state, mask)
                action = tf.argmax(vals, axis=1 if conf.mask else 0, output_type=tf.int32)
            else:
                _, action = sample_action(policy_model, curr_state, mask)
            state, r, t1, t2, step_info = env.step(action[0])
            done = t1 or t2
            reward += r

        if reward < conf.update_dataset_reward_threshold:
            query_set.append(env.query)

        state, info = env.reset()
        reached_end = info["reached_end"]

    env.update_query_set(query_set)
    env.set_queries_random()

