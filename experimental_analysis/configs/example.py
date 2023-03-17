from datetime import datetime
from tensorflow.keras.layers import Dense, Activation, Softmax, concatenate
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model

from src.configuration import PPOConfiguration
from src.envs.join_order.actions import action_regular, action_possible_idx
from src.models.quantum.vqc_layer_tfq import VQC_Layer, layer_ry_rz_cz, encoding_ops_rx

### Configuration
conf = PPOConfiguration()
conf.pg_cost_model = False

# Environment
conf.num_relations = 4
conf.target_num_relations = 4
conf.num_inputs = [conf.num_relations, conf.num_relations, conf.num_relations**2, conf.num_relations**2]
conf.num_actions = conf.num_relations * (conf.num_relations -1)
conf.action_calc = action_regular
conf.mask = True
conf.take_best_threshold = .1
conf.take_best_frequency = 1
conf.update_dataset_reward_threshold = 0.5

conf.gather_selectivity_info = False
conf.multistep = True

# Model
input1 = Input(shape=(conf.num_relations,), name='join_indices')
input2 = Input(shape=(conf.num_relations,), name='cardinalities')
input3 = Input(shape=(conf.num_relations**2), name='tree_structure')
input4 = Input(shape=(conf.num_relations**2), name='join_predicates')
conf.all_inputs = [input1, input2, input3, input4]
conf.concat_ins = concatenate(conf.all_inputs)

# Hyperparameter
conf.total_timesteps = 500000
conf.max_num_steps_per_episode = 4
conf.num_updates = 200
conf.v_coeff = 1.0
conf.ent_coeff = 0.01
conf.clip_ratio = 0.2
conf.train_iterations = 10
conf.save_every = 500

# Hyperparameter subject to search
batchsize = 20
mini_batchsize = 32
lr_start = 8e-5
lr_duration = 0.9
nodes = 128

# Configure params
conf.num_episodes = batchsize
conf.num_updates = 20000//batchsize
conf.validate_every = 500//batchsize
conf.mini_batchsize = mini_batchsize

if lr_duration == -1:
    lr_schedule = lr_start
    lr_schedule_critic = lr_start
else:
    lr_schedule = schedules.PolynomialDecay(lr_start, conf.num_updates * conf.train_iterations * ((conf.max_num_steps_per_episode * conf.num_episodes) // conf.mini_batchsize) * len(conf.policy_model.trainable_variables) * lr_duration, 1e-6, power=0.9)
    lr_schedule_critic = schedules.PolynomialDecay(lr_start, conf.num_updates * conf.train_iterations * ((conf.max_num_steps_per_episode * conf.num_episodes) // conf.mini_batchsize) * len(conf.policy_model.trainable_variables) * lr_duration, 1e-6, power=0.9)

conf.lr = lr_schedule
conf.lr_critic = lr_schedule_critic

# Actor
layer1 = Dense(nodes, activation="relu")(conf.concat_ins)
layer2 = Dense(nodes, activation="relu")(layer1)
layer3 = Dense(conf.num_actions, activation="linear")(layer2)

conf.policy_model = Model(conf.all_inputs, layer3)

# Critic
num_qubits = sum(conf.num_inputs) // conf.num_relations
quantum_critic_layer = VQC_Layer(num_qubits,
                    4,
                    encoding_ops_rx,
                    layer_ry_rz_cz,
                    data_reuploading=False,
                    num_actions=1,
                    incremental_data_uploading=True,
                    num_inputs=sum(conf.num_inputs),
                )(conf.concat_ins)
classical_critic_layer = Dense(1)(quantum_critic_layer)

conf.critic_model = Model(conf.all_inputs, classical_critic_layer)
