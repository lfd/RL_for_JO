import importlib
import sys
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
from datetime import datetime

from src.validate import validate_queries
from src.configuration import PPOConfiguration

def main(config_name, weight_file, k, num_samples):
    config_module = importlib.import_module(f'configs.{config_name}')
    config_path = os.path.relpath(config_module.__file__, f'{os.getcwd()}/configs')
    config = config_module.conf
    config.experiment_name = f'executions/{config_path}'
    config.query_path = f"queries/generated/JOB_splits_rels4/split{k:02d}/train_queries/"
    config.val_query_path = f"queries/generated/JOB_splits_rels4/split{k:02d}/test_queries/"
    config.cost_based = False
    config.timeout = 60
    config.build()
    os.makedirs(config.experiment_name, exist_ok=True)
    os.system(f'cp configs/{config_path} {config.experiment_name}/config.py')

    if isinstance(config, PPOConfiguration):
        validate_queries(config, weight_file, num_samples)
    else:
        exit("Supported algorithm is PPO")


if __name__ == '__main__':
    if len(sys.argv) not in [4, 5]:
        exit('Usage: python execute.py <name-of-config> <path-to-weight-file> <num-samples> [<k>] (Example: python execute.py basic_config logs/model.pkl 1000)')

    config_name = sys.argv[1]
    weight_file = sys.argv[2]
    num_samples = int(sys.argv[3])
    k = int(sys.argv[4]) if len(sys.argv) > 4 else None
    main(config_name, weight_file, k, num_samples)
