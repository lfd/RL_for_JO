from datetime import datetime
from tensorflow.keras.layers import Dense, Activation, Softmax, concatenate
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model

from src.configuration import PPOConfiguration
from src.envs.join_order.actions import action_regular, action_possible_idx
from src.envs.join_order.rewards import reward_mrc_shift
from src.database.database_utils import JO_Setting_Tool

### Configuration
conf = PPOConfiguration()

# Environment
conf.num_relations = 39
num_selection_predicates = 208
conf.num_inputs = [conf.num_relations**2, conf.num_relations**2, num_selection_predicates]
conf.num_actions = conf.num_relations * (conf.num_relations - 1)
conf.action_calc = action_regular
conf.reward_calc = reward_mrc_shift
conf.mask = True

conf.take_best_threshold = 0
conf.take_best_frequency = 20000
conf.update_dataset_reward_threshold = -10
conf.pg_cost_model = True

conf.gather_selectivity_info = False
conf.multistep = False

# Model
input1 = Input(shape=(conf.num_inputs[0],), name='tree_structure')
input2 = Input(shape=(conf.num_inputs[1],), name='join_predicates')
input3 = Input(shape=(conf.num_inputs[2],), name='selection_predicates')

concat_ins = concatenate([input1, input2, input3])

conf.nodes = 128

layer1 = Dense(conf.nodes, activation="relu")(concat_ins)
layer2 = Dense(conf.nodes, activation="relu")(layer1)
layer3 = Dense(conf.num_actions, activation="linear")(layer2)

conf.policy_model = Model([input1, input2, input3], layer3)

critic_layer1 = Dense(conf.nodes, activation="relu")(concat_ins)
critic_layer2 = Dense(conf.nodes, activation="relu")(critic_layer1)
critic_layer3 = Dense(1, activation="linear")(critic_layer2)

conf.critic_model = Model([input1, input2, input3], critic_layer3)

# Hyperparameter
conf.total_timesteps = 500000
conf.num_episodes = 100
conf.max_num_steps_per_episode = 17
conf.num_updates = 200
conf.mini_batchsize = 16
conf.v_coeff = 1.0
conf.ent_coeff = 0.01
conf.clip_ratio = 0.2
conf.train_iterations = 10
conf.validate_every = 10
conf.save_every = 500

