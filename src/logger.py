import tensorflow as tf
from tensorflow.summary import create_file_writer, scalar, histogram
import numpy as np
import csv
import os
import pickle
from math import log, exp

class Logger:
    """
    Logs training progress in Tensorboard
    """

    def __init__(self, log_path, create_tensorboard = True, validation_only = False):

        # Episode statistics
        self.episode = 0
        self.episode_rewards = []
        self.episode_explorations = 0

        self.val_returns = []
        self.val_step = 0

        # Logger
        self.summary_writer = create_file_writer(log_path) if create_tensorboard else None

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        self.csv_log_file_query_val = f"{log_path}/query_val.csv"
        if not os.path.exists(self.csv_log_file_query_val):
            with open(self.csv_log_file_query_val, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "query", "reward", "mrc", "mrt", "geqo_mrc", "costs",
                                 "DP_costs", "geqo_costs", "ex_time", "pg_ex_time", "geqo_ex_time"])

        if not validation_only:
            self.csv_log_file_episode = f"{log_path}/episodes.csv"
            with open(self.csv_log_file_episode, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "length", "return", "exploration"])

            self.csv_log_file_step = f"{log_path}/step.csv"
            with open(self.csv_log_file_step, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "loss", "reward_strength"])

            self.csv_log_file_val_step = f"{log_path}/val_step.csv"
            with open(self.csv_log_file_val_step, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["val_step", "return"])

            self.csv_log_file_batch = f"{log_path}/batch.csv"
            with open(self.csv_log_file_batch, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "avg_rewards", "loss", "mrc", "gmrc", "smrc", "mrt", "gmrt", "smrt", "geqo_mrc", "geqo_gmrc"])

            self.csv_log_file_minibatch_ppo = f"{log_path}/minibatch_ppo.csv"
            with open(self.csv_log_file_minibatch_ppo, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["batch", "iteration", "avg_rewards", "policy_loss", "value_loss", "entropy", "loss"])

            self.csv_log_file_val_avg = f"{log_path}/query_val_avg.csv"
            with open(self.csv_log_file_val_avg, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "mrc", "gmrc", "smrc", "mrt", "gmrt", "smrt", "geqo_mrc", "geqo_gmrc"])

            self.csv_log_file_val_train_avg = f"{log_path}/query_val_train_avg.csv"
            with open(self.csv_log_file_val_train_avg, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "mrc", "gmrc", "smrc", "mrt", "gmrt", "smrt", "geqo_mrc", "geqo_gmrc"])

            self.weight_path = f'{log_path}/weights/'
            os.makedirs(self.weight_path, exist_ok=True)

    def log_transition(self, transition, did_explore):
        self.episode_rewards.append(transition.reward)
        self.episode_explorations += int(did_explore)

        if transition.is_terminal:
            episode_length = len(self.episode_rewards)
            episode_return = sum(self.episode_rewards)
            exploration_freq = self.episode_explorations / episode_length

            with self.summary_writer.as_default():
                scalar('episode/length', episode_length, self.episode)
                scalar('episode/return', episode_return, self.episode)
                scalar('episode/exploration', exploration_freq, self.episode)

            with open(self.csv_log_file_episode, "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode, episode_length, episode_return, exploration_freq])

            self.episode_rewards = []
            self.episode_explorations = 0
            self.episode += 1

    def log_training(self, step, loss, batch):
        # reward_strength = tf.reduce_sum(tf.abs(batch.rewards)) / len(batch.rewards)

        with self.summary_writer.as_default():
            scalar('step/loss', loss, step)
            # scalar('step/reward_strength', reward_strength, step)

        with open(self.csv_log_file_step, "a") as f:
            writer = csv.writer(f)
            # writer.writerow([step, loss, reward_strength])

    def log_validation(self, val_return, grads=None, trainable_variables=None):
        with self.summary_writer.as_default():
            if grads:
                for g, v in zip(grads, trainable_variables):
                    histogram(f'epoch/grads/{v.name}', g, self.val_step)
                    histogram(f'epoch/weights/{v.name}', v, self.val_step)

            tf.summary.scalar('epoch/avg_return', val_return, self.val_step)

        with open(self.csv_log_file_val_step, "a") as f:
            writer = csv.writer(f)
            writer.writerow([self.val_step, val_return])

        self.val_step += 1

    def log_validation_step(self, step, obs, q_values):

        with self.summary_writer.as_default():
            for i, ob in enumerate(obs):
                scalar(f'val_step/obs{i}', ob, self.val_step+step+1)

            for i, q_value in enumerate(q_values):
                scalar(f'val_step/q_value{i}', q_value, self.val_step+step+1)

    def log_batch(self, batch, batch_len, avg_return, sum_rewards = None,
                  costs = None, DP_costs = None,
                  ex_times = None, pg_ex_times = None,
                  geqo_costs = None):

        if costs is None and ex_times is None:
            return
        if costs:
            mrc = self._get_mrc(costs, DP_costs)
            gmrc = self._get_gmrc(costs, DP_costs)
            smrc = self._get_smrc(costs, DP_costs)
            geqo_mrc = self._get_mrc(geqo_costs, DP_costs)
            geqo_gmrc = self._get_gmrc(geqo_costs, DP_costs)
        if ex_times:
            mrt = self._get_mrc(ex_times, pg_ex_times)
            gmrt = self._get_gmrc(ex_times, pg_ex_times)
            smrt = self._get_smrc(ex_times, pg_ex_times)

        with self.summary_writer.as_default():
            scalar('episode/avg_return', avg_return, batch*batch_len)

            if sum_rewards:
                scalar('episode/sum_rewards', sum_rewards, batch*batch_len)
            if costs:
                scalar('episode/mrc', mrc, batch*batch_len)
                scalar('episode/gmrc', gmrc, batch*batch_len)
                scalar('episode/geqo_mrc', geqo_mrc, batch*batch_len)
                scalar('episode/geqo_gmrc', geqo_gmrc, batch*batch_len)
                scalar('episode/smrc', smrc, batch*batch_len)
            if ex_times:
                scalar('episode/mrt', mrt, batch*batch_len)
                scalar('episode/gmrt', gmrt, batch*batch_len)
                scalar('episode/smrt', smrt, batch*batch_len)

        with open(self.csv_log_file_batch, "a") as f:
            mrc = mrc if costs else -1
            gmrc = gmrc if costs else -1
            geqo_mrc = geqo_mrc if costs else -1
            geqo_gmrc = geqo_gmrc if costs else -1
            smrc = smrc if costs else -1
            mrt = mrt if ex_times else -1
            gmrt = gmrt if ex_times else -1
            smrt = smrt if ex_times else -1

            writer = csv.writer(f)
            writer.writerow([batch*batch_len, avg_return, batch_len, mrc, gmrc, smrc, mrt, gmrt, smrt, geqo_mrc, geqo_gmrc])


    def log_minibatch_ppo(self, batch, iteration, avg_return,
                          policy_loss, value_loss, entropy, loss, sum_rewards = None,
                          grads = None,
                          policy_trainable_variables=None, critic_trainable_variables=None):
        with self.summary_writer.as_default():
            # if policy_grads and policy_trainable_variables:
            #     for g, v in zip(policy_grads, policy_trainable_variables):
            #         histogram(f'iteration/policy_grads/{v.name}', g, iteration)
            #         histogram(f'iteration/policy_weights/{v.name}', v, iteration)
            # if value_grads and critic_trainable_variables:
            #     for g, v in zip(value_grads, critic_trainable_variables):
            #         histogram(f'iteration/value_grads/{v.name}', g, iteration)
            #         histogram(f'iteration/value_weights/{v.name}', v, iteration)

            scalar('iteration/policy_loss', policy_loss, iteration)
            scalar('iteration/value_loss', value_loss, iteration)
            scalar('iteration/entropy', entropy, iteration)
            scalar('iteration/loss', loss, iteration)

        with open(self.csv_log_file_minibatch_ppo, "a") as f:
            writer = csv.writer(f)
            writer.writerow([batch, iteration, avg_return, policy_loss,
                             value_loss, entropy, loss])


    def validate_policy_on_query(self, batch, batch_len, query_name, reward,
                                 costs = None, DP_costs = None,
                                 ex_times = None, pg_ex_times = None,
                                 geqo_costs = None, geqo_ex_times = None):

        if costs is None and ex_times is None:
            return
        mrc = -1
        gmrc = -1
        mrt = -1
        gmrt = -1
        geqo_mrc = -1
        geqo_gmrc = -1

        if costs:
            mrc = self._get_mrc([costs], [DP_costs])
            gmrc = self._get_gmrc([costs], [DP_costs])
            geqo_mrc = self._get_mrc([geqo_costs], [DP_costs])
            geqo_gmrc = self._get_gmrc([geqo_costs], [DP_costs])
        if ex_times:
            mrt = self._get_mrc([ex_times], [pg_ex_times])
            gmrt = self._get_gmrc([ex_times], [pg_ex_times])

        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                scalar(f'episode/reward/query{query_name}', reward, batch*batch_len)

                if costs:
                    scalar(f'episode/validation/mrc/query{query_name}', mrc, batch*batch_len)
                    scalar(f'episode/validation/gmrc/query{query_name}', gmrc, batch*batch_len)
                    scalar(f'episode/validation/geqo_mrc/query{query_name}', geqo_mrc, batch*batch_len)
                    scalar(f'episode/validation/geqo_gmrc/query{query_name}', geqo_gmrc, batch*batch_len)
                if ex_times:
                    scalar(f'episode/validation/mrt/query{query_name}', mrt, batch*batch_len)
                    scalar(f'episode/validation/gmrt/query{query_name}', gmrt, batch*batch_len)

        with open(self.csv_log_file_query_val, "a") as f:
            writer = csv.writer(f)
            writer.writerow([batch*batch_len, query_name, reward, mrc, mrt, geqo_mrc, costs,
                             DP_costs, geqo_costs, ex_times, pg_ex_times, geqo_ex_times])

    def log_validation_avg(self, batch, batch_len, avg_reward,
                           costs, DP_costs,
                           ex_times, pg_ex_times, geqo_costs,
                           training = False):

        if costs is None and ex_times is None:
            return

        mrc = self._get_mrc(costs, DP_costs)
        gmrc = self._get_gmrc(costs, DP_costs)
        geqo_mrc = self._get_mrc(geqo_costs, DP_costs)
        geqo_gmrc = self._get_gmrc(geqo_costs, DP_costs)
        smrc = self._get_smrc(costs, DP_costs)
        mrt = self._get_mrc(ex_times, pg_ex_times)
        gmrt = self._get_gmrc(ex_times, pg_ex_times)
        smrt = self._get_smrc(ex_times, pg_ex_times)

        log_name = "trainings" if training else "validations"

        with self.summary_writer.as_default():
            scalar(f'episode/{log_name}/avg_reward', avg_reward, batch*batch_len)

            scalar(f'episode/{log_name}/mrc', mrc, batch*batch_len)
            scalar(f'episode/{log_name}/gmrc', gmrc, batch*batch_len)
            scalar(f'episode/{log_name}/smrc', smrc, batch*batch_len)
            scalar(f'episode/{log_name}/mrt', mrt, batch*batch_len)
            scalar(f'episode/{log_name}/gmrt', gmrt, batch*batch_len)
            scalar(f'episode/{log_name}/smrt', smrt, batch*batch_len)
            scalar(f'episode/{log_name}/geqo_mrc', geqo_mrc, batch*batch_len)
            scalar(f'episode/{log_name}/geqo_gmrc', geqo_gmrc, batch*batch_len)

        filename = self.csv_log_file_val_train_avg if training else \
                self.csv_log_file_val_avg

        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow([batch*batch_len, avg_reward, mrc, gmrc, smrc, mrt, gmrt, smrt, geqo_mrc, geqo_gmrc])


    def save_model_parameters(self, parameters, batch, critic_model_weights = None):
        with open(f'{self.weight_path}step_{batch:04d}.pkl', 'wb') as f:
            pickle.dump(parameters, f)
        if critic_model_weights:
            with open(f'{self.weight_path}critic_step_{batch:04d}.pkl', 'wb') as f:
                pickle.dump(critic_model_weights, f)

    def _get_mrc(self, costs, DP_costs):
        rel_costs = []
        ctr = 0
        for c, p in zip(costs, DP_costs):
            if p == 0:
                if c == 0:
                    p = c = 1
                else:
                    ctr += 1
                    continue
            if c == 0:
                p = c = 1
            rel_costs.append(c/p)
        for _ in range(ctr):
            if len(rel_costs) > 0:
                rel_costs.append(max(rel_costs))
            else:
                rel_costs.append(100)
        return sum(rel_costs) / len(rel_costs)

    def _get_gmrc(self, costs, DP_costs):
        rel_costs = []
        ctr = 0
        for c, p in zip(costs, DP_costs):
            if p == 0:
                if c == 0:
                    p = c = 1
                else:
                    ctr += 1
                    continue
            if c == 0:
                p = c = 1
            rel_costs.append(log(c/p))
        for _ in range(ctr):
            if len(rel_costs) > 0:
                rel_costs.append(max(rel_costs))
            else:
                rel_costs.append(log(100))
        return exp(sum(rel_costs) / len(rel_costs))

    def _get_smrc(self, costs, DP_costs):
        return sum(costs) / sum(DP_costs)
