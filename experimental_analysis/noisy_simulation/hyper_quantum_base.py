import importlib
import sys
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
from datetime import datetime
import copy
from sys import argv
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

sys.path.insert(0, os.getcwd())

from src.validate import validate_queries
from src.algorithms.policy_gradient.algorithm import train as train_pg
from src.algorithms.policy_gradient.ppo.algorithm import train as train_ppo
from src.configuration import DQNConfiguration, PGConfiguration, PPOConfiguration
from src.models.quantum.vqc_layer_qiskit import VQC_Layer, layer_ry_rz_cz, encoding_ops_rx

def main(config_name, data_reuploading, num_layers, k, depol_error_prob, weight_file):
    config_module = importlib.import_module(f'configs.{config_name}')
    config_path = os.path.relpath(__file__, f'{os.getcwd()}')[:-3]
    config = config_module.conf

    config.query_path = f"queries/generated/JOB_splits_rels4/split{k:02d}/train_queries/"
    config.val_query_path = f"queries/generated/JOB_splits_rels4/split{k:02d}/test_queries/"

    num_qubits = sum(config.num_inputs) // config.num_relations
    quantum_policy_layer = VQC_Layer(num_qubits,
                       num_layers,
                       encoding_ops_rx,
                       layer_ry_rz_cz,
                       data_reuploading=data_reuploading,
                       num_actions=config.num_actions,
                       incremental_data_uploading=True,
                       num_inputs=sum(config.num_inputs),
                       depol_error_prob=depol_error_prob,
                    )(config.concat_ins)
    classical_policy_layer = Dense(config.num_actions)(quantum_policy_layer)

    config.policy_model = Model(config.all_inputs, classical_policy_layer)

    quantum_critic_layer = VQC_Layer(num_qubits,
                       num_layers,
                       encoding_ops_rx,
                       layer_ry_rz_cz,
                       data_reuploading=data_reuploading,
                       num_actions=1,
                       incremental_data_uploading=True,
                       num_inputs=sum(config.num_inputs),
                       depol_error_prob=depol_error_prob,
                    )(config.concat_ins)
    classical_critic_layer = Dense(1)(quantum_critic_layer)

    config.critic_model = Model(config.all_inputs, classical_critic_layer)

    config.experiment_name = f'logs/validation/{config_path}/rels{config.num_relations}/depol_error_prob{str(depol_error_prob).replace(".", "")}/num_layers{num_layers}/{"data_reupl" if data_reuploading else "no_data_reupl"}/k{k}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    print(config.experiment_name)
    os.makedirs(config.experiment_name, exist_ok=True)
    config.build()

    validate_queries(config, weight_file, 1)

if __name__ == '__main__':
    config_name = argv[1]
    data_reuploading = bool(int(argv[2]))
    num_layers = int(argv[3])
    k = int(argv[4])
    depol_error_prob = float(argv[5])
    weight_file = argv[6]

    main(config_name, data_reuploading, num_layers, k, depol_error_prob, weight_file)
