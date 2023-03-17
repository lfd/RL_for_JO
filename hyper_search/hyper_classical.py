import importlib
import sys
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
from datetime import datetime
from sys import argv
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

sys.path.insert(0, os.getcwd())

from src.algorithms.policy_gradient.ppo.algorithm import train as train_ppo

def main(config_name, lr_start, multistep, selpreds, lr_duration, take_best_frequency, data_reuploading, num_layers, nodes, batchsize, mini_batchsize, k):
    config_module = importlib.import_module(f'configs.{config_name}')
    config_path = os.path.relpath(__file__, f'{os.getcwd()}')[:-3]
    config = config_module.conf
    config.query_path = f"queries/generated/JOB_splits_rels4/split{k:02d}/train_queries/"
    config.val_query_path = f"queries/generated/JOB_splits_rels4/split{k:02d}/test_queries/"

    config.num_episodes = batchsize
    config.num_updates = 20000//batchsize
    config.validate_every = 500//batchsize
    config.mini_batchsize = mini_batchsize

    if lr_duration == -1:
        lr_schedule = lr_start
        lr_schedule_critic = lr_start
    else:
        lr_schedule = schedules.PolynomialDecay(lr_start, config.num_updates * config.train_iterations * ((config.max_num_steps_per_episode * config.num_episodes) // config.mini_batchsize) * len(config.policy_model.trainable_variables) * lr_duration, 1e-6, power=0.9)
        lr_schedule_critic = schedules.PolynomialDecay(lr_start, config.num_updates * config.train_iterations * ((config.max_num_steps_per_episode * config.num_episodes) // config.mini_batchsize) * len(config.policy_model.trainable_variables) * lr_duration, 1e-6, power=0.9)

    config.lr = lr_schedule
    config.lr_critic = lr_schedule_critic

    config.multistep = multistep
    config.gather_selectivity_info = selpreds
    config.take_best_frequency = take_best_frequency

    layer1 = Dense(nodes, activation="relu")(config.concat_ins)
    layer2 = Dense(nodes, activation="relu")(layer1)
    layer3 = Dense(config.num_actions, activation="linear")(layer2)

    config.policy_model = Model(config.all_inputs, layer3)

    critic_layer1 = Dense(nodes, activation="relu")(config.concat_ins)
    critic_layer2 = Dense(nodes, activation="relu")(critic_layer1)
    critic_layer3 = Dense(1, activation="linear")(critic_layer2)

    config.critic_model = Model(config.all_inputs, critic_layer3)

    config.experiment_name = f'logs/xval/{config_path}/rels{config.num_relations}/num_layers{num_layers}/{"data_reupl" if data_reuploading else "no_data_reupl"}/num_episodes{batchsize}/mini_batchsize{mini_batchsize}/best_frequency{config.take_best_frequency}/lr_start{str(lr_start).replace(".", "")}/lr_duration{str(lr_duration).replace(".", "")}/take_best_threshold{str(config.take_best_threshold).replace(".", "")}/{"multistep" if multistep else "no_multistep"}/{"selpreds" if selpreds else "no_selpreds"}/k{k}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    print(config.experiment_name)
    os.makedirs(config.experiment_name, exist_ok=True)
    config.build()

    train_ppo(config)



if __name__ == '__main__':
    config_name = argv[1]
    lr_start = float(argv[2])
    multistep = bool(int(argv[3]))
    selpreds = bool(int(argv[4]))
    lr_duration = float(argv[5])
    take_best_frequency = int(argv[6])
    data_reuploading = 0
    num_layers = 0
    nodes = int(argv[9])
    batch = int(argv[10])
    mbatch = int(argv[11])
    k = int(argv[12])

    main(config_name, lr_start, multistep, selpreds, lr_duration, take_best_frequency, data_reuploading, num_layers, nodes, batch, mbatch, k)
