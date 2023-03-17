import importlib
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import schedules
tf.get_logger().setLevel('ERROR')
import os
from datetime import datetime
import copy
from sys import argv

from src.algorithms.dqn.algorithm import train as train_dqn
from src.algorithms.policy_gradient.algorithm import train as train_pg
from src.algorithms.policy_gradient.ppo.algorithm import train as train_ppo
from src.configuration import DQNConfiguration, PGConfiguration, PPOConfiguration

def main(config_name, lr_start, multistep, selpreds, lr_duration, take_best_frequency, batchsize, mini_batchsize, threshold, k):
    config_module = importlib.import_module(f'configs.{config_name}')
    config_path = os.path.relpath(config_module.__file__, f'{os.getcwd()}/configs')
    config = config_module.conf
    config.k_fold = k
    config.num_episodes = batchsize
    config.num_updates = 20000//batchsize
    config.validate_every = 500//batchsize
    config.mini_batchsize = mini_batchsize

    if lr_duration == -1:
        lr_schedule = lr_start
    else:
        lr_schedule = schedules.PolynomialDecay(lr_start, config.num_updates * config.train_iterations * ((config.max_num_steps_per_episode * config.num_episodes) // config.mini_batchsize) * len(config.policy_model.trainable_variables) * lr_duration, 1e-8, power=0.9)
    config.lr = lr_schedule
    config.lr_out = lr_schedule
    config.multistep = multistep
    config.gather_selectivity_info = selpreds
    config.take_best_frequency = take_best_frequency
    config.take_best_threshold = threshold
    config.experiment_name = f'logs/xval_low_lr/{config_path}/num_episodes{batchsize}/mini_batchsize{mini_batchsize}/best_frequency{config.take_best_frequency}/nodes{config.nodes}/lr_start{str(lr_start).replace(".", "")}/lr_duration{str(lr_duration).replace(".", "")}/{"multistep" if multistep else "no_multistep"}/take_best_threshold{str(threshold).replace(".", "")}/{"selpreds" if selpreds else "no_selpreds"}/k{k}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    print(config.experiment_name)
    os.makedirs(config.experiment_name, exist_ok=True)
    config.build()

    train_ppo(config)


if __name__ == '__main__':
    config_name = sys.argv[1]
    lr_start = float(argv[2])
    multistep = bool(int(argv[3]))
    selpreds = bool(int(argv[4]))
    lr_duration = float(argv[5])
    take_best_frequency = int(argv[6])
    batch = int(argv[7])
    mbatch = int(argv[8])
    threshold = float(argv[9])
    k = int(argv[10])

    main(config_name, lr_start, multistep, selpreds, lr_duration, take_best_frequency, batch, mbatch, threshold, k)

