import numpy as np

from .policies import Policy


def validate(env, policy: Policy, num_steps: int = None, num_trials=None, logger=None) -> float:
    """
    Validates a policy for a certain number of trials or steps.
    """
    returns = []
    total_reward = 0

    obs, _ = env.reset()
    done = False

    if num_steps:
        for i in range(num_steps):

            action, q_value = policy(obs)
            obs, reward, t1, t2, _ = env.step(action)
            done = t1 or t2
            total_reward += reward

            if done:
                returns.append(total_reward)
                total_reward = 0
                env.reset()
                done = False

    elif num_trials:
        for i in range(num_trials):
            done=False

            step=0
            while not done:
                obs_base=obs
                action, q_value = policy(obs)
                obs, reward, t1, t2, _ = env.step(action)
                done = t1 or t2
                total_reward += reward

                if logger and not isinstance(obs, list):
                    logger.log_validation_step(i+step, obs_base, q_value)

                step+=1

            returns.append(total_reward)
            total_reward = 0
            env.reset()
            done = False

    return np.mean(returns) if len(returns) > 0 else total_reward
