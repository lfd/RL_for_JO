'''
Join Order MDP environment

Original implementation by:
Guo Xintong
https://github.com/GUOXINTONG/rejoin
Commit: 02365ab0f1cb2e0019be062babf0d66aaa93c4eb
File: src/state.py

Modified.
'''

import numpy as np
from math import log
import itertools
import moz_sql_parser
import copy

class StateVector_ReJoin:
    def __init__(self, query, relations, attributes, num_relations, reduced_actions):

        self.relations = relations
        self.attributes = attributes
        self.num_relations = num_relations
        self.reduced_actions = reduced_actions

        self.query = query

        self.indices = copy.copy(query.indices)
        self.num_non_empty_rows = len(self.indices)

        # State Representation (to be fed in the NN)
        self.tree_structure = self.extract_tree_structure()

        if self.num_relations is None:
            self.selection_predicates = self.extract_selection_predicates()
        else:
            self.indices = [v / len(self.query.database.tables) for v in self.indices ]
            self.indices += [-0.5] * (self.num_relations - len(self.indices))
            self.selectivities = query.selectivities
            self.selectivities += [-0.5] * (self.num_relations - len(self.selectivities))

        self.join_predicates = self.extract_join_predicates()

        self.valid_actions = self.get_valid_actions()

    def extract_tree_structure(self):
        # Initial State is a rxr identity matrix
        if self.num_relations is None:
            graph = np.zeros((len(self.relations), len(self.relations)), dtype=np.float32)

            for idx in self.indices:
                graph[idx][idx] = 1

        else:
            graph = np.zeros((self.num_relations, self.num_relations), dtype=np.float32)

            for i in range(len(self.indices)):
                graph[i][i] = 1

        return graph

    def extract_join_predicates(self):

        if self.num_relations is None:
            if self.query.gather_sel_info:
                graph = np.ones((len(self.relations), len(self.relations)), dtype=np.float32)
            else:
                graph = np.zeros((len(self.relations), len(self.relations)), dtype=np.float32)
        else:
            if self.query.gather_sel_info:
                graph = np.ones((self.num_relations, self.num_relations), dtype=np.float32)
            else:
                graph = np.zeros((self.num_relations, self.num_relations), dtype=np.float32)

        for t1, t2 in self.query.joined_attrs.keys():
            selectivity = self.query.joined_attrs[(t1,t2)].selectivity
            val = 1 if not self.query.gather_sel_info or selectivity is None else selectivity
            if self.num_relations is None:
                graph[self.relations.index(t1)][self.relations.index(t2)] = val
            else:
                idx_t1 = self.query.alias_list.index(t1)
                idx_t2 = self.query.alias_list.index(t2)
                graph[idx_t1][idx_t2] = val

        return graph

    def extract_selection_predicates(self):
        all_attrs = list(self.attributes.keys())

        sel_predicate_vector = np.zeros(len(all_attrs), dtype=np.float32)
        for attr in list(self.query.attr_vals.keys()) + list(self.query.join_attr_vals.keys()):
            sel_predicate_vector[all_attrs.index(attr)] = 1.

        return np.array(sel_predicate_vector)

    def vectorize(self, tree_structure = None):
        if tree_structure is None:
            tree_structure = self.tree_structure
        tree_structure = np.array(tree_structure, dtype=np.float32).flatten()
        join_predicates = np.array(self.join_predicates, dtype=np.float32).flatten()

        if self.num_relations is None:
            selection_predicates = np.array(self.selection_predicates, dtype=np.float32)
            return [tree_structure, join_predicates, selection_predicates]
        else:
            indices = np.array(self.indices, dtype=np.float32)
            selectivities = np.array(self.selectivities, dtype=np.float32)
            return [indices, selectivities, tree_structure, join_predicates]

    def get_valid_actions(self):
        actions = []
        sel_val = 1 if self.query.gather_sel_info else 0

        if self.num_relations is None:
            num_relations = len(self.relations)
        else:
            num_relations = self.num_relations

        for i in range(0, num_relations):
            for idx1, val1 in enumerate(self.tree_structure[i]):
                if val1 == 0:
                    continue
                for j in range(i + 1, num_relations):
                    for idx2, val2 in enumerate(self.tree_structure[j]):
                        if val2 == 0 or self.join_predicates[idx1][idx2] == sel_val:
                            continue
                        if (i, j) not in actions:
                            actions.append((i, j))
                        if not self.reduced_actions and \
                                (j, i) not in actions:
                            actions.append((j, i))

        return actions

    def set_next_state(self, action_pair):

        self.tree_structure = self.get_next_state(action_pair)
        self.num_non_empty_rows -= 1
        self.valid_actions = self.get_valid_actions()

    def get_next_state(self, action_pair):

        if self.num_relations is None:
            num_relations = len(self.relations)
        else:
            num_relations = self.num_relations

        if self.reduced_actions and action_pair[0] > action_pair[1]:
            action_pair = (action_pair[1], action_pair[0])

        tree_structure = self._join_subtrees(action_pair[0], action_pair[1], num_relations)

        return tree_structure

    def get_flattened_next_state(self, action_pair):
        tree_structure = self.get_next_state(action_pair)
        return self.vectorize(tree_structure)

    def _join_subtrees(self, t1, t2, num_relations):

        # [0  1  0  0  0] JOIN
        # [0  0  1  0  0] =
        # [0 .5 .5  0  0]

        tree_structure = copy.deepcopy(self.tree_structure)

        s1 = tree_structure[t1]
        s2 = tree_structure[t2]

        for i in range(0, num_relations):
            if s1[i] != 0:
                s1[i] = s1[i] / 2
            elif s2[i] != 0:
                s1[i] = s2[i] / 2
                s2[i] = 0

        return tree_structure


    def get_tree_structure_rows(self, action_pair):
        row_idx1, row_idx2 = -1, -1
        num_relations = len(self.tree_structure[0])

        for row in range(num_relations):
            if row_idx1 > -1 and row_idx2 > -1:
                break
            for col in range(num_relations):
                if self.tree_structure[row][col] > 0:
                    if col == action_pair[0]:
                        row_idx1 = row
                    elif col == action_pair[1]:
                        row_idx2 = row

        return (row_idx1, row_idx2)

    def is_terminal(self):
        return self.num_non_empty_rows <= 2
