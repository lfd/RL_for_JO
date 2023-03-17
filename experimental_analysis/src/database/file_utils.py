import os
import csv
import ast

from .database_utils import JoinInfo

def get_cardinality_from_file_for_alias(alias, alias_sel_file):
    if not create_cardinality_file(alias_sel_file):
        with open(alias_sel_file, "r") as f:
            reader = csv.reader(f)
            for i, (a, unfiltered_cardinality, filtered_cardinality) in enumerate(reader):
                if i == 0:
                    continue
                if alias == a:
                    return int(unfiltered_cardinality), int(filtered_cardinality)
    return None, None

def create_cardinality_file(alias_sel_file):
    if not os.path.exists(alias_sel_file):
        os.makedirs(os.path.dirname(alias_sel_file), exist_ok=True)
        with open(alias_sel_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["alias", "unfiltered_cardinality", "filtered_cardinality"])
        return True
    return False

def store_join_selectivities(table_left, alias_left, attr_left, table_right, alias_right, attr_right, selectivity, join_sel_file):
    with open(join_sel_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([table_left, alias_left, attr_left, table_right, alias_right, attr_right, selectivity])

def store_order(order, cost_pg, cost_DP, order_file):
    with open(order_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([order, cost_pg, cost_DP])

def store_cost_for_order(order, cost, order_file):
    with open(order_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([order, cost])

def store_db_info(table, attribute_name, n_distinct, data_type, cardinality, min, max, db_info_file):
    with open(db_info_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([table, attribute_name, n_distinct, data_type, cardinality, min, max])

def store_card_for_joined_aliases(rels, cardinality, join_alias_card_file):
    with open(join_alias_card_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([rels, cardinality])

def store_card_for_alias(alias, unfiltered_cardinality, filtered_cardinality, alias_sel_file):
    with open(alias_sel_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([alias, unfiltered_cardinality, filtered_cardinality])

def load_costs_for_orders(query_cost_info_file):
    orders = dict()
    if not os.path.exists(query_cost_info_file):
        os.makedirs(os.path.dirname(query_cost_info_file), exist_ok=True)
        with open(query_cost_info_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["order", "costs"])
    else:
        with open(query_cost_info_file, "r") as f:
            reader = csv.reader(f)
            for i, (str_order, cost) in enumerate(reader):
                if i == 0:
                    continue
                o = ast.literal_eval(str_order)
                orders[o] = float(cost)
    return orders

def load_db_info(db_info_file):
    info = []
    if not os.path.exists(db_info_file):
        os.makedirs(os.path.dirname(db_info_file), exist_ok=True)
        with open(db_info_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["table", "attribute", "num_distinct", "data_type", "cardinality", "min", "max"])
    else:
        with open(db_info_file, "r") as f:
            reader = csv.reader(f)
            for i, (table, attribute, num_distinct,  data_type, cardinality, min, max) in enumerate(reader):
                if i == 0:
                    continue
                info.append((table, attribute, num_distinct, data_type, cardinality, min, max))
    return info

def load_order(order_file):
    if not os.path.exists(order_file):
        os.makedirs(os.path.dirname(order_file), exist_ok=True)
        with open(order_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["order", "cost_pg", "cost_DP"])
    else:
        with open(order_file, "r") as f:
            reader = csv.reader(f)
            for i, (str_aliases, cost_pg, cost_DP) in enumerate(reader):
                if i == 0:
                    continue
                order = ast.literal_eval(str_aliases)
                return order, float(cost_pg), float(cost_DP)
    return None, None, None

def load_DP_order(DP_order_file):
    order = None
    min = None
    if not os.path.exists(DP_order_file):
        os.makedirs(os.path.dirname(DP_order_file), exist_ok=True)
        with open(DP_order_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["order", "cost_DP"])
    else:
        with open(DP_order_file, "r") as f:
            reader = csv.reader(f)
            for i, (str_aliases, cost_DP) in enumerate(reader):
                if i == 0:
                    continue
                cost_DP = float(cost_DP)
                if min is None or cost_DP < min:
                    order = ast.literal_eval(str_aliases)
                    min = cost_DP
    return order, min

def load_cardinalities_for_rels(join_alias_card_file):
    cards = dict()
    if not os.path.exists(join_alias_card_file):
        os.makedirs(os.path.dirname(join_alias_card_file), exist_ok=True)
        with open(join_alias_card_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["aliases", "cardinality"])
    else:
        with open(join_alias_card_file, "r") as f:
            reader = csv.reader(f)
            for i, (str_aliases, cardinality) in enumerate(reader):
                if i == 0:
                    continue
                rels = ast.literal_eval(str_aliases)
                cards[rels] = int(cardinality)
    return cards


def load_join_selectivities_from_file(join_sel_file):
    joined_attrs = dict()
    if not os.path.exists(join_sel_file):
        os.makedirs(os.path.dirname(join_sel_file), exist_ok=True)
        with open(join_sel_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["table_left", "alias_left", "attr_left", "table_right", "alias_right", "attr_right", "selectivity"])
    else:
        with open(join_sel_file, "r") as f:
            reader = csv.reader(f)
            for i, (table_left, alias_left, attr_left, table_right, alias_right, attr_right, selectivity) in enumerate(reader):
                if i == 0:
                    continue
                joined_attrs[(alias_left, alias_right)] = JoinInfo(table_left, alias_left, attr_left, table_right, alias_right, attr_right)
                joined_attrs[(alias_right, alias_left)] = JoinInfo(table_right, alias_right, attr_right, table_left, alias_left, attr_left)
                joined_attrs[(alias_left, alias_right)].set_selectivity(selectivity)
                joined_attrs[(alias_right, alias_left)].set_selectivity(selectivity)

    return joined_attrs

