import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import GlorotUniform
from abc import ABC
import numpy as np
import random

from src.envs.join_order.environment import JoinOrdering
from src.envs.join_order.states import StateVector_ReJoin
from src.envs.join_order.actions import action_regular
from src.envs.join_order.rewards import reward_ratio_psql
from src.database.database_utils import JO_Setting_Tool as jst
from src.database.database import Database
from src.database.sql_info import split_dataset


class Configuration(ABC):
    """
    Abstract configuration template with default values.
    A configuration can be changed by accessing its public Attributes, e.g.:

    # change optimizer and learning rate
    config = DQNConfiguration()
    config.optimizer = Adam(learning_rate=0.001)

    For a list of available options, please refer to the __init__() method.

    """
    def __init__(self, seed = 27):
        self.seed = seed
        self.set_seed()

        # Basic Hyperparameters for both DQN and policy gradient
        self.batch_size = 32
        self.gamma = 0.99
        self.save_every = 100
        self.num_episodes = 100

        # Optimizers
        self.optimizer_in = None
        self.optimizer_out = None
        self.optimizer_critic = None
        self.optimizer_critic_in = None
        self.optimizer_critic_out = None

        self.loss = losses.MSE

        # Environment
        self.target_num_relations = None
        self.jo_setting_tool = jst.PG_HINT_PLAN
        self.query_path = "queries/jo-bench/"
        self.val_query_path = "queries/jo-bench/"
        self.base_query_path = "queries/jo-bench/"
        self.test_queries = []
        self.mask = False
        self.cost_based = True
        self.cost_training = True
        self.k_fold = None
        self.multistep = False
        self.gather_selectivity_info = True
        self.use_geqo = False
        self.reduced_actions = False
        self.lr = 1e-3
        self.lr_out = 1e-3
        self.lr_critic = None
        self.lr_critic_out = None
        self.weight_decay = 0
        self.optimizer = None
        self.wrapper = None
        self.load_costs_from_file = False

        # curriculum learning
        self.num_curriculum = None
        self.curriculum_interval = None

        self.take_best_threshold = 0.3
        self.take_best_frequency = 10
        self.take_best_warmup = 0

        self.pg_cost_model = True

        self.update_dataset_every = 200
        self.update_dataset_reward_threshold = 0

        self.generate_queries_during_training = False
        self.search_exhaustive = False

        self.timeout = 300

        self.depol_error_prob = 0.

    def set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)

    def build(self):
        database = Database(collect_db_info=True, jo_setting_tool=self.jo_setting_tool,
                            base_query_path=self.base_query_path, use_geqo=self.use_geqo,
                            timeout=self.timeout)

        trainset, testset = split_dataset(self.query_path,
                                          database,
                                          self.target_num_relations,
                                          self.test_queries,
                                          self.k_fold,
                                          self.val_query_path,
                                          self.gather_selectivity_info,
                                          self.pg_cost_model,
                                          self.target_num_relations is not None,
                                          self.generate_queries_during_training,
                                          self.load_costs_from_file,
                                          self.search_exhaustive,
                                          not self.cost_based,
                                          )

        self.env = JoinOrdering(
            state_calc=self.state_calc,
            action_calc=self.action_calc,
            reward_calc=self.reward_calc,
            queries=trainset,
            database=database,
            target_num_relations=self.target_num_relations,
            cost_based=self.cost_based,
            cost_training=self.cost_training,
            multistep=self.multistep,
            reduced_actions=self.reduced_actions,
            num_curriculum=self.num_curriculum,
            curriculum_interval=self.curriculum_interval,
        )

        self.val_env = JoinOrdering(
            state_calc=self.state_calc,
            action_calc=self.action_calc,
            reward_calc=self.reward_calc,
            queries=testset,
            database=database,
            target_num_relations=self.target_num_relations,
            cost_based=self.cost_based,
            cost_training=self.cost_training,
            validation=True,
            multistep=self.multistep,
            reduced_actions=self.reduced_actions,
        )

        self.incr_train_env = JoinOrdering(
            state_calc=self.state_calc,
            action_calc=self.action_calc,
            reward_calc=self.reward_calc,
            queries=trainset,
            database=database,
            target_num_relations=self.target_num_relations,
            cost_based=self.cost_based,
            cost_training=self.cost_training,
            validation=True,
            multistep=self.multistep,
            reduced_actions=self.reduced_actions,
        )


        if self.wrapper:
            self.env = self.wrapper(self.env)
            self.val_env = self.wrapper(self.val_env)

        if self.lr_critic is None:
            self.lr_critic = self.lr

        if self.lr_critic_out is None:
            self.lr_critic_out = self.lr_out

        if self.optimizer is None:
            self.optimizer = Adam(learning_rate=self.lr, decay=self.weight_decay)
            self.optimizer_critic = Adam(learning_rate=self.lr_critic, decay=self.weight_decay)

        if self.lr != self.lr_out:
            self.optimizer_out = Adam(learning_rate=self.lr_out, decay=self.weight_decay)

        if self.lr_critic != self.lr_critic_out:
            self.optimizer_critic_out = Adam(learning_rate=self.lr_critic_out, decay=self.weight_decay)


class DQNConfiguration(Configuration):
    """
    Configuration template with default values for DQN
    A configuration can be changed by accessing its public Attributes.

    For a list of available options, please refer to the __init__() method.
    """
    def __init__(self):
        super(DQNConfiguration, self).__init__()

        # Basic Hyperparameters for DQN
        self.experiment_name = "DQN"
        self.train_after = 100
        self.train_every = 1
        self.update_every = 30
        self.validate_every = 100
        self.replay_capacity=50000
        self.num_val_trials = 10
        self.num_val_steps = None
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_duration=30000
        self.prio = False
        self.prefill_buffer = False

        # Model
        num_inputs=4
        num_actions=2
        self.policy_model = Sequential([
            Dense(8, kernel_initializer=GlorotUniform(27)),
            Activation('relu'),
            Dense(num_actions, kernel_initializer=GlorotUniform(27)),
            Activation('linear'),
        ])
        self.target_model = Sequential([
            Dense(8, kernel_initializer=GlorotUniform(27)),
            Activation('relu'),
            Dense(num_actions, kernel_initializer=GlorotUniform(27)),
            Activation('linear'),
        ])
        self.policy_model.build((1, 4))
        self.target_model.build((1, 4))
        self.target_model.set_weights(self.policy_model.get_weights())

        self.action_calc = action_regular
        self.state_calc=StateVector_ReJoin
        self.reward_calc=reward_ratio_psql

    def get_optimizers(self):
        """
        Assigns the model parameters to each optimizer.

        Returns:
            optimizers: A list of tensorflow optimizers
            trainable_variables: corresponding parameter list
        """
        num_vars_policy = len(self.policy_model.trainable_variables)

        if self.optimizer_in:
            if self.optimizer_out:
                policy_optimizers = [self.optimizer_in] + [self.optimizer] * (num_vars_policy-2) + [self.optimizer_out]
            else:
                policy_optimizers = [self.optimizer_in] + [self.optimizer] * (num_vars_policy-1)
        else:
            if self.optimizer_out:
                policy_optimizers = [self.optimizer] * (num_vars_policy-1) + [self.optimizer_out]
            else:
                policy_optimizers = [self.optimizer] * num_vars_policy

        trainable_variables = self.policy_model.trainable_variables

        return policy_optimizers, trainable_variables

class PGConfiguration(Configuration):
    """
    Configuration template with default values for policy gradient
    A configuration can be changed by accessing its public Attributes.

    For a list of available options, please refer to the __init__() method.
    """
    def __init__(self, seed = 27):
        super(PGConfiguration, self).__init__(seed)

        # Basic Hyperparameters for policy gradient
        self.experiment_name = "policy_gradient"
        self.total_timesteps = 50000
        self.max_num_steps_per_episode = 200
        self.num_updates = 1024
        self.validate_every = 10
        self.save_every = 10

        # Model
        num_inputs=4
        num_actions=2
        self.policy_model = Sequential([
            Dense(8, kernel_initializer=GlorotUniform(27)),
            Activation('relu'),
            Dense(num_actions, kernel_initializer=GlorotUniform(27)),
        ])
        self.policy_model.build((1, 4))

        # Environment
        self.action_calc = action_regular
        self.state_calc=StateVector_ReJoin
        self.reward_calc=reward_ratio_psql


class PPOConfiguration(PGConfiguration):
    """
    Configuration template with default values for proximatal policy optimisation
    A configuration can be changed by accessing its public Attributes.

    For a list of available options, please refer to the __init__() method.
    """
    def __init__(self, seed = 27):
        super(PPOConfiguration, self).__init__(seed)

        # Basic Hyperparameters for ppo
        self.experiment_name = "ppo"
        self.mini_batchsize = 64
        self.v_coeff = 1.
        self.ent_coeff = 0.01
        self.clip_ratio=0.2
        self.train_iterations=4

        # Model
        num_inputs=4
        self.num_actions=2
        self.policy_model = Sequential([
            Dense(32, kernel_initializer=GlorotUniform(27)),
            Activation('relu'),
            Dense(self.num_actions, kernel_initializer=GlorotUniform(27)),
        ])
        self.policy_model.build((1, num_inputs))

        self.critic_model = Sequential([
            Dense(32, kernel_initializer=GlorotUniform(27)),
            Activation('relu'),
            Dense(1, kernel_initializer=GlorotUniform(27))
        ])
        self.critic_model.build((1, num_inputs))

    def build(self):
        super().build()

        if self.optimizer_critic is None:
            self.optimizer_critic = Adam(learning_rate=self.lr, decay=self.weight_decay)

    def get_optimizers(self):
        """
        Assigns the model parameters to each optimizer.

        Returns:
            optimizers: A list of tensorflow optimizers
            trainable_variables: corresponding parameter list
        """
        num_vars_policy = len(self.policy_model.trainable_variables)
        num_vars_critic = len(self.critic_model.trainable_variables)

        if self.optimizer_in:
            if self.optimizer_out:
                policy_optimizers = [self.optimizer_in] * 2 + [self.optimizer] * (num_vars_policy-4) + [self.optimizer_out] * 2
            else:
                policy_optimizers = [self.optimizer_in] * 2 + [self.optimizer] * (num_vars_policy-2)
        else:
            if self.optimizer_out:
                policy_optimizers = [self.optimizer] * (num_vars_policy-2) + [self.optimizer_out] * 2
            else:
                policy_optimizers = [self.optimizer] * num_vars_policy

        if self.optimizer_in:
            if self.optimizer_out:
                critic_optimizers = [self.optimizer_critic_in] * 2 + [self.optimizer_critic] * (num_vars_critic-4) + [self.optimizer_critic_out] * 2
            else:
                critic_optimizers = [self.optimizer_critic_in] * 2 + [self.optimizer_critic] * (num_vars_critic-2)
        else:
            if self.optimizer_out:
                critic_optimizers = [self.optimizer_critic] * (num_vars_critic-2) + [self.optimizer_critic_out] * 2
            else:
                critic_optimizers = [self.optimizer_critic] * num_vars_critic

        trainable_variables = self.policy_model.trainable_variables
        if len(policy_optimizers) != num_vars_policy:
            policy_optimizers = policy_optimizers[len(policy_optimizers)-num_vars_policy:]
        if len(critic_optimizers) != num_vars_critic:
            critic_optimizers = critic_optimizers[len(critic_optimizers)-num_vars_critic:]
        optimizers = policy_optimizers

        variable_names = [v.name for v in self.policy_model.trainable_variables]

        for i, v in enumerate(self.critic_model.trainable_variables):
            if v.name not in variable_names:
                trainable_variables.append(v)
                optimizers.append(critic_optimizers[i])

        return optimizers, trainable_variables
