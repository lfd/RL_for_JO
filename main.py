import importlib
import sys
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
from datetime import datetime

from src.algorithms.dqn.algorithm import train as train_dqn
from src.algorithms.policy_gradient.algorithm import train as train_pg
from src.algorithms.policy_gradient.ppo.algorithm import train as train_ppo
from src.configuration import DQNConfiguration, PGConfiguration, PPOConfiguration

def main(config_name, weight_file):
    config_module = importlib.import_module(f'configs.{config_name}')
    config_path = os.path.relpath(config_module.__file__, f'{os.getcwd()}/configs')
    config = config_module.conf
    config.experiment_name = f'logs/{config_path}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(config.experiment_name, exist_ok=True)
    os.system(f'cp configs/{config_path} {config.experiment_name}/config.py')
    config.build()

    if isinstance(config, DQNConfiguration):
        train_dqn(config, weight_file)
    elif isinstance(config, PPOConfiguration):
        train_ppo(config, weight_file)
    elif isinstance(config, PGConfiguration):
        train_pg(config, weight_file)
    else:
        exit("Supported algorithms are Deep Q-Learning and Policy Gradient")


if __name__ == '__main__':
    if len(sys.argv) not in [2,3]:
        exit('Usage: python main.py <name-of-config> [<path-to-weight-file>] (Example: python main.py example)')

    config_name = sys.argv[1]
    weight_file = sys.argv[2] if len(sys.argv) == 3 else None
    main(config_name, weight_file)
