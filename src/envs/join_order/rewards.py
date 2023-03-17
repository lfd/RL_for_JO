import numpy as np
from math import log10, log

def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def reward_reciprocal(reward, query_file=None, cost_based=None):
    return 1 / reward

def reward_reciprocal_log(reward, query_file=None, cost_based=None):
    return 1 / log10(reward)

def reward_plain(reward, query_file=None, cost_based=None):
    return reward

def reward_negative(reward, query_file=None, cost_based=None):
    return -reward

def reward_ratio_psql(reward, query, cost_based):

    if cost_based:
        psql_reward = query.get_pg_cost()
    else:
        psql_reward = query.get_pg_execution_time()

    if psql_reward == 0:
        psql_reward = 1
    ret_val = reward / psql_reward
    ret_val = log(ret_val+1) if ret_val > 0 else ret_val
    ret_val = ret_val if ret_val < 5 else 5

    return -ret_val

def reward_relative(reward, query, cost_based):

    if cost_based:
        psql_reward = query.get_pg_cost()
    else:
        psql_reward = query.get_pg_execution_time()

    if reward == 0:
        reward = 1

    ret_val = psql_reward / reward

    return ret_val


def reward_mrc_shift(reward, query, cost_based):

    if cost_based:
        psql_reward = query.get_pg_cost()
    else:
        psql_reward = query.get_pg_execution_time()

    if reward == 0:
        reward = 1

    if psql_reward == 0:
        psql_reward = 1

    ret_val = reward / psql_reward
    ret_val = log(ret_val+1) if ret_val > 0 else ret_val
    ret_val = ret_val if ret_val < 4 else 4

    return -ret_val + log(2) + 1




