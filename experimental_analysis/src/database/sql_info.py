import moz_sql_parser
import csv
import random
from collections import defaultdict
from math import log2, log, inf
from pathlib import Path
import copy
from itertools import combinations, permutations

from . import database_utils as utils
from . import file_utils as fu

class Query_Info:
    def __init__(self, query_path: str,
                 filename: str,
                 database,
                 gather_sel_info=True,
                 pg_cost_model=True,
                 indices_from_table=True,
                 generate_queries=False,
                 load_costs_from_file = False,
                 search_exhaustive = True,
                 extime = False):
        self.load_costs_from_file = load_costs_from_file
        self.postgres_execution_time = None
        self.geqo_execution_time = None
        self.planning_time = None
        self.filename = filename
        self.database = database
        self.execution_times_for_orders = dict()
        self.extime = extime
        self.previous_orders = []
        self.gather_sel_info = gather_sel_info
        self.pg_cost_model = pg_cost_model
        self.query_cost_info_file = f"{query_path}/costs/{filename[:-4]}.csv"
        self.DP_order_file = f"{query_path}/orders_DP/{filename[:-4]}.csv"
        if pg_cost_model:
            self.join_alias_card_file = f"{query_path}/join_cardinalities/estimated/{filename[:-4]}.csv"
            self.alias_sel_file = f"{query_path}/cardinalities/estimated/{filename[:-4]}.csv"
        else:
            self.join_alias_card_file = f"{query_path}/join_cardinalities/actual/{filename[:-4]}.csv"
            self.alias_sel_file = f"{query_path}/cardinalities/actual/{filename[:-4]}.csv"
        self.join_sel_file = f"{query_path}/join_selectivities/{filename[:-4]}.csv"
        self.cost_for_order = fu.load_costs_for_orders(self.query_cost_info_file) if load_costs_from_file else dict()
        self.cardinalities_for_rels = fu.load_cardinalities_for_rels(self.join_alias_card_file)
        self.query_path = query_path
        self.generate_queries = generate_queries
        self.search_exhaustive = search_exhaustive

        with open(f'{query_path}/{filename}', 'r') as f:
            self.sql = f.read()

        self.ast = moz_sql_parser.parse(self.sql)

        self.aliases = dict()
        for v in self.ast["from"]:
            self.aliases[v["name"]] = v["value"]

        self.aliases = dict(sorted(self.aliases.items()))
        self.alias_list = list(self.aliases.keys())

        self.num_relations = len(self.aliases)

        # Help structures for query reconstruction
        self.alias_to_relations = dict()
        for alias in self.aliases:
            self.alias_to_relations[alias] = [alias]

        self.joined_attrs = fu.load_join_selectivities_from_file(self.join_sel_file)
        self.attr_vals, self.join_attr_vals = self._get_attrs()
        if indices_from_table:
            self.indices = [self.database.tables.index(table) for alias, table in self.aliases.items()]
        else:
            self.indices = [self.database.relations.index(alias) for alias, table in self.aliases.items()]

        if indices_from_table or not self.pg_cost_model:
            self.selectivities, self.cardinalities_full = self._get_filtered_cardinalities(self.attr_vals)

        self.pg_order, self.postgres_cost = self.get_postgres_order_and_cost()
        self.geqo_order, self.geqo_cost = self.get_postgres_order_and_cost(geqo=True)
        if self.pg_cost_model:
            self.best_order, self.DP_cost = self.pg_order, None
        else:
            self.best_order, self.DP_cost = self.get_DP_order_and_cost()
        if self.extime:
            self.execution_times_for_orders[utils.to_tuple(self.pg_order)] = self.get_pg_execution_time()
            self.execution_times_for_orders[utils.to_tuple(self.geqo_order)] = self.get_geqo_execution_time()

    def get_pg_execution_time(self, fetch_new=False):
        if self.postgres_execution_time == None or fetch_new:
            constructed_query = self.database.query_parser.construct_query(self, self.pg_order)
            self.planning_time, self.postgres_execution_time = \
                    self.database.get_query_time(constructed_query, True, self.database.jo_setting_tool)
        return self.postgres_execution_time

    def get_geqo_execution_time(self, fetch_new=False):
        if self.geqo_execution_time == None or fetch_new:
            constructed_query = self.database.query_parser.construct_query(self, self.geqo_order)
            _, self.geqo_execution_time = self.database.get_query_time(constructed_query, True, self.database.jo_setting_tool)
        return self.geqo_execution_time

    def get_pg_planning_time(self, fetch_new=False):
        if self.planning_time == None or fetch_new:
            self.planning_time, self.postgres_execution_time = \
                    self.database.get_query_time(self.sql, False, self.database.jo_setting_tool)
        return self.planning_time

    def get_pg_cost(self):
        if self.postgres_cost == None:
            _, self.postgres_cost = self.get_postgres_order_and_cost()
        return self.postgres_cost

    def get_geqo_cost(self):
        if self.geqo_cost == None:
            _, self.geqo_cost = self.get_postgres_order_and_cost(geqo=True)
        return self.geqo_cost

    def get_DP_cost(self):
        if self.pg_cost_model:
            return self.get_pg_cost()
        if self.DP_cost == None:
            _, self.DP_cost = self.get_DP_order_and_cost()
        return self.DP_cost

    def get_mrc_for_partial_orders(self, orders):
        cost_sum = 0
        current_cost = 0
        num_orders = 0

        for partial_order in orders:

            if isinstance(partial_order, str):
                continue

            tuple_order = utils.to_tuple(partial_order)
            constructed_query = None

            if tuple_order not in self.cost_for_order:
                if self.pg_cost_model:
                    constructed_query = self.database.query_parser.construct_partial_query(self, partial_order)
                    cost = self.database.optimizer_cost(constructed_query, True, self.database.jo_setting_tool)
                    if self.load_costs_from_file:
                        fu.store_cost_for_order(tuple_order, cost, self.query_cost_info_file)
                else:
                    cost = self.get_cost_out(partial_order)
                self.cost_for_order[tuple_order] = cost

            if constructed_query is None:
                constructed_query = self.database.query_parser.construct_partial_query(self, partial_order)

            current_cost += self.cost_for_order[tuple_order]
            cost_sum += self.cost_for_order[tuple_order]

            for prev_order in self.previous_orders:
                if prev_order in tuple_order or prev_order == tuple_order:
                    current_cost -= self.cost_for_order[prev_order]

            num_orders += 1

        baseline_cost = self.get_DP_cost()

        if current_cost < 0:
            return 0, cost_sum
        else:
            if baseline_cost == 0:
                baseline_cost = 1
            mrc = current_cost / baseline_cost
        ret_val = mrc if mrc < self.num_relations - 1 else self.num_relations - 1
        ret_val = (-ret_val + 2) / (self.num_relations -1)

        return ret_val, cost_sum

    def get_cost_for_order(self, order):
        tuple_order = utils.to_tuple(order)
        if not tuple_order in self.cost_for_order:
            if self.pg_cost_model:
                constructed_query = self.database.query_parser.construct_query(self, order)
                cost = self.database.optimizer_cost(constructed_query, True, self.database.jo_setting_tool)
                if self.load_costs_from_file:
                    fu.store_cost_for_order(order, cost, self.query_cost_info_file)
            else:
                cost = self.get_cost_out(order)

            self.cost_for_order[tuple_order] = cost

        return self.cost_for_order[tuple_order]

    def update_best_order(self, order):
        tuple_order = utils.to_tuple(order)
        if self.extime:
            if self.best_order is None or \
                self.execution_times_for_orders[tuple_order] < self.execution_times_for_orders[utils.to_tuple(self.best_order)]:
                self.best_order = order
        else:
            if self.best_order is None or \
                self.cost_for_order[tuple_order] < self.cost_for_order[utils.to_tuple(self.best_order)]:
                self.best_order = order
                if not self.pg_cost_model:
                    self.DP_cost = self.cost_for_order[tuple_order]
                    fu.store_cost_for_order(order, self.cost_for_order[tuple_order], self.DP_order_file)

    def get_best_order(self):
        return self.best_order

    def get_execution_time_for_order(self, order, fetch_new = False):
        order = utils.to_tuple(order)
        if not order in self.execution_times_for_orders or fetch_new:
            constructed_query = self.database.query_parser.construct_query(self, order)
            _, ex_time = self.database.get_query_time(constructed_query, True, self.database.jo_setting_tool)
            self.execution_times_for_orders[order] = ex_time
        return self.execution_times_for_orders[order]

    def get_postgres_order_and_cost(self, geqo = False):
        self.database.execute_prefix(False, self.database.jo_setting_tool, use_geqo = geqo)
        if self.database.jo_setting_tool == utils.JO_Setting_Tool.NATIVE:
            cost_pg = self.database.optimizer_cost(self.sql, force_order = False, use_geqo = geqo)
            cost_out = -1
            order = None
        else:
            plan = self.database.get_query_plan(self.sql)
            order = utils.get_jo_for_plan(plan)
            cost_pg = plan['Total Cost']
            cost_out = self.get_cost_out(order)
        cost = cost_pg if self.pg_cost_model else cost_out
        if order is not None:
            self.cost_for_order[utils.to_tuple(order)] = cost
        return order, cost

    def get_DP_order_and_cost(self):
        dp_order, cost_out = fu.load_DP_order(self.DP_order_file)
        if dp_order is None:
            if self.search_exhaustive:
                dp_order, cost_out = self.search_optimal_order_DP()
                fu.store_cost_for_order(dp_order, cost_out, self.DP_order_file)
            else:
                dp_order, cost_out = self.get_postgres_order_and_cost()
        if cost_out == 0:
            cost_out = 1
        self.cost_for_order[utils.to_tuple(dp_order)] = cost_out
        return dp_order, cost_out

    def search_optimal_order_DP(self):
        foreign_keys = list(self.joined_attrs.keys())
        todo = copy.copy(self.alias_list)
        k = self.num_relations
        best_tree = { (r,): r for r in todo }
        for i in range(2, k + 1):
            for s in combinations(todo, i):
                min_cost = inf
                for o in permutations(s):
                    for j in range(len(o)):
                        left, right = best_tree.get(o[:j]), best_tree.get(o[j:])
                        if left is None or right is None:
                            continue
                        order = [best_tree[o[:j]], best_tree[o[j:]]]
                        if utils.is_cross_free_join(order, foreign_keys):
                            rels = tuple(sorted(o))
                            cost = self.get_cost_out(order)
                            if cost < min_cost:
                                best_tree[rels] = order
                                min_cost = cost
        return best_tree[tuple(self.alias_list)], min_cost

    def get_cost_out(self, order):
        if isinstance(order, str):
            return 0

        left, right = order
        cost_left = self.get_cost_out(left)
        cost_right = self.get_cost_out(right)

        return self.get_cardinality_for_order(order) + \
                cost_left + cost_right

    def get_cardinality_for_order(self, order):
        if isinstance(order, str):
            return self.cardinalities_full[order][0]
        rels = sorted(utils.flatten_nested_str_list(order))
        rels = tuple(rels)
        if rels not in self.cardinalities_for_rels:
            query = self.database.query_parser.construct_partial_query(self, order)
            cardinality = self.database.get_count_result(query, self.database.jo_setting_tool, estimation=self.pg_cost_model)
            if cardinality is None: # Timeout reached
                cardinality = self.get_cardinality_for_order(order[0]) * self.get_cardinality_for_order(order[1])
            else:
                fu.store_card_for_joined_aliases(rels, cardinality, self.join_alias_card_file)
            self.cardinalities_for_rels[rels] = cardinality
        return self.cardinalities_for_rels[rels]

    def _get_filtered_cardinalities(self, attr_vals):
        aliases_attributes = defaultdict(list)
        selectivities = []
        cardinalities_filtered_unfiltered = dict()

        for attr_name, predicates in attr_vals.items():
            alias = attr_name.split('.')[0]
            aliases_attributes[alias].extend(predicates)

        for alias, table in self.aliases.items():
            unfiltered_cardinality, filtered_cardinality = fu.get_cardinality_from_file_for_alias(alias, self.alias_sel_file)
            if unfiltered_cardinality is None:
                unfiltered_cardinality, filtered_cardinality = self._get_unfiltered_and_filtered_cardinalities(table, alias, aliases_attributes[alias])
                fu.store_card_for_alias(alias, unfiltered_cardinality, filtered_cardinality, self.alias_sel_file)
            selectivities.append(filtered_cardinality / unfiltered_cardinality)
            cardinalities_filtered_unfiltered[alias] = (filtered_cardinality, unfiltered_cardinality)

        return selectivities, cardinalities_filtered_unfiltered

    def _get_unfiltered_and_filtered_cardinalities(self, table, alias, predicates):

        unfiltered_cardinality = self.database.get_num_tuples(table, alias, [], estimation=self.pg_cost_model)
        filtered_cardinality = self.database.get_num_tuples(table, alias, predicates, estimation=self.pg_cost_model)

        return unfiltered_cardinality, filtered_cardinality

    def _get_join_selectivity(self, join_info):
        table1 = self.database.relations_tables[join_info.alias1]
        table2 = self.database.relations_tables[join_info.alias2]

        num_tuples = self.database.get_num_tuples_for_join(table1, table2, join_info.attr1, join_info.attr2)

        cardinality1 = self.database.attributes[f"{join_info.alias1}.{join_info.attr1}"].cardinality
        cardinality2 = self.database.attributes[f"{join_info.alias2}.{join_info.attr2}"].cardinality

        return num_tuples / (cardinality1 * cardinality2)

    def _get_attrs(self):

        attr_vals = defaultdict(list)
        join_attr_vals = defaultdict(list)

        for v in self.ast["where"]["and"]:
            for k in v:
                if k == "eq" and isinstance(v[k][0], str) and isinstance(v[k][1], str):
                    alias_left, attr1 = v[k][0].split(".")
                    alias_right, attr2 = v[k][1].split(".")

                    table_left = self.database.relations_tables[alias_left]
                    table_right = self.database.relations_tables[alias_right]

                    join_attr1 = utils.JoinInfo(table_left, alias_left, attr1, table_right, alias_right, attr2)
                    join_attr2 = utils.JoinInfo(table_right, alias_right, attr2, table_left, alias_left, attr1)

                    if (alias_left, alias_right) not in self.joined_attrs or \
                            (alias_right, alias_left) not in self.joined_attrs:
                        if self.gather_sel_info:
                            sel = self._get_join_selectivity(join_attr1)
                            join_attr1.set_selectivity(sel)
                            join_attr2.set_selectivity(sel)
                            fu.store_join_selectivities(table_left, alias_left, attr1, table_right, alias_right, attr2, sel, self.join_sel_file)
                        self.joined_attrs[(alias_left, alias_right)] = join_attr1
                        self.joined_attrs[(alias_right, alias_left)] = join_attr2

                    join_attr_vals[v[k][0]].append(f"{v[k][0]} == {v[k][1]}")
                    join_attr_vals[v[k][1]].append(f"{v[k][0]} == {v[k][1]}")
                else:
                    av = self._get_attr_vals(v, k)
                    for key, val in av.items():
                        attr_vals[key].extend(val)

        return attr_vals, join_attr_vals

    def _get_attr_vals(self, v, k):
        attr_vals = defaultdict(list)
        sign_dict = utils.get_operator_map()

        if k == "or":
            av_or = defaultdict(list)
            for l in v[k]:
                for m in l:
                    av = self._get_attr_vals(l, m)
                    for key, val in av.items():
                        av_or[key].extend(val)
            for key, vals in av_or.items():
                attr_vals[key].extend([f'({" OR ".join(vals)})'])

        elif k == "missing":
            attr_vals[v[k]].append(f'{v[k]} IS NULL')

        elif k == "exists":
            attr_vals[v[k]].append(f'{v[k]} IS NOT NULL')

        elif k == "between":
            if isinstance(v[k][1], dict):
                val1 = f"\'{v[k][1]['literal']}\'"
            else:
                val1 = v[k][1]

            if isinstance(v[k][2], dict):
                val2 = f"\'{v[k][2]['literal']}\'"
            else:
                val2 = v[k][2]

            attr_vals[v[k][0]].append(f'{v[k][0]} BETWEEN {val1} AND {val2}')

        elif k == "in":
            if isinstance(v[k][1], dict):
                vals = v[k][1]["literal"]
                if isinstance(vals, list):
                    vals = [f"\'{v}\'" for v in vals]
            else:
                vals = v[k][1]

            if isinstance(vals, str):
                valstr = f"\'{vals}\'"
            elif len(vals) == 1:
                valstr = vals[0]
            else:
                valstr = ", ".join(vals)

            attr_vals[v[k][0]].append(f'{v[k][0]} IN ({valstr})')

        elif isinstance(v[k][0], str):
            if isinstance(v[k][1], dict):
                val = f"\'{v[k][1]['literal']}\'"
            else:
                val = v[k][1]
            attr_vals[v[k][0]].append(f'{v[k][0]} {sign_dict[k]} {val}')

        elif isinstance(v[k][1], str):
            if isinstance(v[k][0], dict):
                val = f"\'{v[k][0]['literal']}\'"
            else:
                val = v[k][0]
            attr_vals[v[k][1]].append(f'{v[k][1]} {sign_dict[k]} {val}')

        return attr_vals

    def reset_previous_orders(self):
        self.previous_orders = []

    def set_previous_orders(self, previous_orders):
        self.previous_orders = []

        for partial_order in previous_orders:

            if isinstance(partial_order, str):
                continue

            partial_order = utils.to_tuple(partial_order)
            self.previous_orders.append(partial_order)

def split_dataset(query_path: str,
                  database,
                  target_num_relations = None,
                  validation_queries = [],
                  k_fold = None,
                  validation_path = None,
                  gather_sel_info = True,
                  pg_cost_model = True,
                  indices_from_table = True,
                  generate_queries = False,
                  load_costs_from_file = False,
                  search_exhaustive = True,
                  extime = False,
                  ):

    query_path = Path(query_path)
    if validation_path is not None:
        validation_path = Path(validation_path) 

    train_queries = []
    test_queries = []

    with open(f"{str(query_path)}/data.csv", 'r') as data_file:
        raw_data = list(csv.reader(data_file, delimiter=','))

    if validation_path and validation_path != query_path:
        with open(f"{str(validation_path)}/data.csv", 'r') as data_file:
            raw_val_data = list(csv.reader(data_file, delimiter=','))

        for filename, num_relations, planning_time, execution_time, cost_dp, cost_geqo in raw_val_data[1:]:
            if target_num_relations is None or int(num_relations) <= target_num_relations:
                test_queries.append(Query_Info(validation_path, filename, database, gather_sel_info, pg_cost_model, indices_from_table, generate_queries, load_costs_from_file, search_exhaustive, extime))

    for filename, num_relations, planning_time, execution_time, cost_dp, cost_geqo in raw_data[1:]:
        if target_num_relations is None or int(num_relations) <= target_num_relations:
            query_partly_val = False
            if filename in validation_queries or query_partly_val:
                test_queries.append(Query_Info(query_path, filename, database, gather_sel_info, pg_cost_model, indices_from_table, generate_queries, load_costs_from_file, search_exhaustive, extime))
            else:
                train_queries.append(Query_Info(query_path, filename, database, gather_sel_info, pg_cost_model, indices_from_table, generate_queries, load_costs_from_file, search_exhaustive, extime))

    num_queries = len(train_queries)
    if k_fold is not None:
        random.shuffle(train_queries)
        splits = [train_queries[s : s + num_queries//10] for s in range(0, num_queries, num_queries//10)]
        if len(splits[-1]) < num_queries//10:
            for i, query in enumerate(splits[-1]):
                splits[i%(len(splits)-1)].append(query)
            splits = splits[:-1]

        test_queries.extend(splits[k_fold])
        splits.remove(splits[k_fold])
        train_queries = [q for s in splits for q in s]

    return train_queries, test_queries


def get_queries_incremental(queries):

    sorted_queries = sorted(queries, key = lambda x: x.filename)

    for q in sorted_queries:
        yield q

    return None


def get_queries_random(queries, num_curriculum, idx):
    if num_curriculum is not None and idx < num_curriculum and not len(queries) < num_curriculum:
        sorted_queries = sorted(queries, key = lambda x: x.num_relations)

        curriculum = []
        curriculum_len = len(queries) // num_curriculum

        for i in range(num_curriculum-1):
            curriculum.append(sorted_queries[i*curriculum_len : (i+1)*curriculum_len])

        curriculum.append(sorted_queries[(i+1)*curriculum_len:])

        q = [query for c in curriculum[:idx+1] for query in c]
    else:
        q = queries

    while(True):
        yield q[random.randint(0, len(q)-1)]

    return None

