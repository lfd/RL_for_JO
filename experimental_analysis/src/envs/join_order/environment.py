import numpy as np
import math
import itertools
from copy import copy

from src.database.database import Database
from src.database.query_parser import Query_Parser
from src.database.sql_info import Query_Info, get_queries_random, get_queries_incremental

class JoinOrdering:
    def __init__(
        self,
        state_calc,
        action_calc,
        reward_calc,
        queries,
        database,
        cost_based = True,
        cost_training = True,
        target_num_relations = None,
        validation = False,
        multistep = False,
        reduced_actions = False,
        num_curriculum = None,
        curriculum_interval = None,
    ):
        self.validation = validation
        self.multistep = multistep

        self.database = database

        self.cost_based = cost_based
        self.cost_training = cost_training
        self.state_calc = state_calc
        self.action_calc = action_calc
        self.reward_calc = reward_calc
        self.reduced_actions = reduced_actions

        self.target_num_relations = target_num_relations

        # curriculum learning parameters
        self.num_curriculum = num_curriculum
        self.curriculum_interval = curriculum_interval
        self.query_idx = 0

        self.base_queries = self.queries = queries
        self.generate_queries = self._get_query_generator_fkt()

        self.query = None
        self.state_vector = None
        self.memory_actions = []

        if self.target_num_relations is None:
            self.num_relations = len(self.database.relations)
        else:
            self.num_relations = target_num_relations

        self.all_combinations = list(itertools.combinations(np.arange(self.num_relations), 2))

        if not reduced_actions:
            self.all_combinations = sorted(self.all_combinations + \
                    list(itertools.combinations(np.arange(self.num_relations)[::-1], 2)))

        self.hashv = " "

    def is_terminal(self):
        return self.state_vector.is_terminal()

    def reset(self, seed=0, options = {}):
        """
        Reset environment and setup for new episode.
        Returns:
            initial state of reset environment.
        """

        # update curriculum
        if self.curriculum_interval is not None and self.query_idx % self.curriculum_interval == 0 \
                and self.query_idx != 0:
            self.generate_queries = self._get_query_generator_fkt()

        # Create a new initial state
        self.query = next(self.generate_queries, None)

        if self.query is None:
            self.generate_queries = self._get_query_generator_fkt()
            self.hashv = " "
            return self.state_vector.vectorize(), {"query_file": None, "reached_end": True}

        print("File Name: " + self.query.filename)

        self.memory_actions = []

        self.state_vector = self.state_calc(
            self.query, self.database.relations, self.database.attributes,
            self.target_num_relations, self.reduced_actions
        )

        # Get reward and process terminal & next state.
        return_state = self.state_vector.vectorize()

        self.query_idx += 1

        return return_state, {"query_file": self.query.filename, "reached_end": False}

    def step(self, action):
        """
        Executes action, observes next state(s) and reward.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (next state, bool indicating terminal, reward)
        """
        info = dict()
        possible_actions = self.state_vector.valid_actions  # [(0,1), (1,0), (1,2), (2,1)]

        action_pair, reward_tmp = self.action_calc(action, possible_actions, self.all_combinations)

        if action_pair is not None:
            terminal = self.is_terminal()

            if self.reduced_actions:
                action_options = [action_pair, action_pair[::-1]]
            else:
                action_options = [action_pair]

            min_plain_costs = math.inf
            min_plain_reward = -math.inf

            if terminal:
                for rp in action_options:
                    ordering, _ = self._get_order(action_pair)
                    if self.multistep:
                        plain_reward, costs = self._get_reward_for_partial_order([ordering])
                        if costs < min_plain_costs:
                            min_plain_costs = costs
                            min_plain_reward = plain_reward
                            final_rp = rp
                            final_ordering = ordering
                    else:
                        plain_reward = costs = self._get_cost_or_execution_time(ordering)
                        if costs < min_plain_costs:
                            min_plain_costs = costs
                            min_plain_reward = plain_reward
                            final_rp = rp
                            final_ordering = ordering

                self.query.update_best_order(final_ordering)

                if self.multistep:
                    reward = min_plain_reward
                else:
                    reward = self.reward_calc(min_plain_reward, self.query, self.cost_based)

                info["cost"] = self.query.get_cost_for_order(final_ordering)
                info["DP_cost"] = self.query.get_DP_cost()
                info["geqo_cost"] = self.query.get_geqo_cost()
                if not self.cost_based:
                    info["execution_time"] = self.query.get_execution_time_for_order(final_ordering, self.cost_training)
                    info["pg_execution_time"] = self.query.get_pg_execution_time(fetch_new=self.cost_training)
                    info["geqo_execution_time"] = self.query.get_geqo_execution_time(fetch_new=self.cost_training)

                self.query.reset_previous_orders()

            elif reward_tmp:
                reward = reward_tmp
            else:
                if self.multistep or self.reduced_actions:
                    choosen_order = None
                    for rp in action_options:
                        _, partial_order = self._get_order(rp)
                        tmp_reward, costs = self._get_reward_for_partial_order(partial_order)
                        if costs < min_plain_costs:
                            min_plain_costs = costs
                            min_plain_reward = tmp_reward
                            final_rp = rp
                            choosen_order = partial_order
                    self.query.set_previous_orders(choosen_order)
                    reward = min_plain_reward if self.multistep else 0
                else:
                    final_rp = action_pair
                    reward = 0.

            self.state_vector.set_next_state(action_pair)
            self.memory_actions.append(final_rp)

        else:
            terminal = True
            reward = reward_tmp

        return_state = self.state_vector.vectorize()
        # print(f'Reward: {reward}')

        return return_state, reward, terminal, False, info

    def _get_reward_for_partial_order(self, partial_order):
        return self.query.get_mrc_for_partial_orders(partial_order)

    def _get_cost_or_execution_time(self, order):
        """Reward is given for the final state."""

        if self.cost_training:
            return self.query.get_cost_for_order(order)
        else:
            return self.query.get_execution_time_for_order(order)

    def _get_order(self, final_action_pair):
        if self.target_num_relations is None:
            join_ordering = self.state_vector.relations.copy()
        else:
            join_ordering = self.query.alias_list.copy()

        final_ordering = []

        for action_pair in self.memory_actions + [final_action_pair]:
            small_idx = min(action_pair[0], action_pair[1]) if self.reduced_actions else action_pair[0]
            big_idx = max(action_pair[0], action_pair[1]) if self.reduced_actions else action_pair[1]
            join_ordering[small_idx] = [
                join_ordering[action_pair[0]],
                join_ordering[action_pair[1]],
            ]

            final_ordering = join_ordering[small_idx]

        return final_ordering, join_ordering

    def get_best_action(self, idx):
        order = self.query.get_best_order()
        action_list = self._actions_from_order(order)
        action_pair = action_list[idx]
        return self.all_combinations.index(action_pair)

    def _actions_from_order(self, order):

        if self.target_num_relations is None:
            aliases = self.state_vector.relations.copy()
        else:
            aliases = self.query.alias_list.copy()

        actions, _ = self._rec_action_from_order(order, aliases)
        return actions

    def _rec_action_from_order(self, order, aliases):
        left, right = order
        actions = []

        if isinstance(left, list):
            a, aliases = self._rec_action_from_order(left, aliases)
            actions.extend(a)
        if isinstance(right, list):
            a, aliases = self._rec_action_from_order(right, aliases)
            actions.extend(a)

        idx_left, idx_right = aliases.index(left), aliases.index(right)
        actions.append((idx_left, idx_right))
        aliases[idx_left] = [left, right]
        return actions, aliases

    def get_mask(self):
        possible_actions = self.state_vector.valid_actions  # [(0,1), (1,0), (1,2), (2,1)]
        mask_dim = self.num_relations*(self.num_relations-1)

        if self.reduced_actions:
            mask_dim //= 2

        mask = np.zeros(mask_dim, dtype=np.float32)

        for action_pair in possible_actions:
            idx = self.all_combinations.index(action_pair)
            mask[idx] = 1

        return mask

    def _get_query_generator_fkt(self):

        if self.validation:
            return get_queries_incremental(self.queries)
        else:
            curr_idx = self.query_idx // self.curriculum_interval if self.curriculum_interval is not None else None
            return get_queries_random(self.queries, self.num_curriculum, curr_idx)

    def get_all_next_states(self):
        possible_actions = self.state_vector.valid_actions
        next_states = [self.state_vector.get_flattened_next_state(pair) for pair in possible_actions]
        return next_states

    def get_query_hash(self):
        return self.query.sql + self._hashv

    def update_query_set(self, query_set):
        self.queries = query_set
        self.generate_queries = self._get_query_generator_fkt()

    def set_queries_incremental(self):
        self.generate_queries = get_queries_incremental(self.base_queries)

    def set_queries_random(self):
        curr_idx = self.query_idx // self.curriculum_interval if self.curriculum_interval is not None else None
        self.generate_queries = get_queries_random(self.queries, self.num_curriculum, curr_idx)
