from datetime import datetime
from tensorflow.keras.layers import Dense, Activation, Softmax, concatenate
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model

from src.configuration import PPOConfiguration
from src.envs.join_order.actions import action_regular

### Configuration
conf = PPOConfiguration()
conf.pg_cost_model = False

# Environment
conf.num_relations = 17
conf.target_num_relations = 17
conf.num_inputs = [conf.num_relations, conf.num_relations, conf.num_relations**2, conf.num_relations**2]
conf.num_actions = conf.num_relations * (conf.num_relations -1)
conf.action_calc = action_regular
conf.mask = True
conf.take_best_threshold = .1
conf.take_best_frequency = 1
conf.update_dataset_reward_threshold = 0.5

# Model
conf.nodes = 384

input0 = Input(shape=(conf.num_inputs[0]), name='table_indices')
input1 = Input(shape=(conf.num_inputs[1]), name='filter_selectivities')
input2 = Input(shape=(conf.num_inputs[2]), name='tree_structure')
input3 = Input(shape=(conf.num_inputs[3]), name='join_predicates')

concat_ins = concatenate([input0, input1, input2, input3])

layer1 = Dense(conf.nodes, activation="relu")(concat_ins)
layer2 = Dense(conf.nodes, activation="relu")(layer1)
layer3 = Dense(conf.num_actions, activation="linear")(layer2)

conf.policy_model = Model([input0, input1, input2, input3], layer3)

critic_layer1 = Dense(conf.nodes, activation="relu")(concat_ins)
critic_layer2 = Dense(conf.nodes, activation="relu")(critic_layer1)
critic_layer3 = Dense(1, activation="linear")(critic_layer2)

conf.critic_model = Model([input0, input1, input2, input3], critic_layer3)

# Hyperparameter
conf.total_timesteps = 500000
conf.num_episodes = 100
conf.max_num_steps_per_episode = 17
conf.num_updates = 200
conf.mini_batchsize = 16
conf.train_iterations = 10
conf.validate_every = 10
conf.save_every = 5000
conf.clip_ratio = 0.2
conf.v_coeff = 1.
conf.ent_coeff = 0.01

