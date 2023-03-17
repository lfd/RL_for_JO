from datetime import datetime
from tensorflow.keras.layers import Dense, Activation, Softmax, concatenate
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model

from src.configuration import PPOConfiguration
from src.envs.join_order.actions import action_regular, action_possible_idx

### Configuration
conf = PPOConfiguration()

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
conf.num_episodes = 100
conf.max_num_steps_per_episode = 17
conf.num_updates = 200
conf.mini_batchsize = 8
conf.v_coeff = 1.0
conf.ent_coeff = 0.01
conf.clip_ratio = 0.2
conf.train_iterations = 10
conf.validate_every = 10
conf.save_every = 500

