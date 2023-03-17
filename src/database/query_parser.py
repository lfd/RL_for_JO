from copy import copy
import os

from src.database.sql_info import Query_Info
from src.database.database_utils import JO_Setting_Tool as jst
import src.database.database_utils as utils
import src.database.file_utils as fu

class Query_Parser:

    def __init__(self, join_setting_tool: jst, attrs):
        self.counter = 0
        self.join_setting_tool = join_setting_tool
        self.attrs = attrs

    def construct_partial_query(self, query: Query_Info, join_ordering):
        if self.join_setting_tool == jst.PG_HINT_PLAN:
            return self.construct_partial_query_pg_hint_plan(query, join_ordering)
        else:
            return self.construct_partial_query_on_jo(query, join_ordering)

    def construct_partial_query_on_jo(self, query: Query_Info, partial_order):
        rels = utils.flatten_nested_str_list(partial_order)

        aliases = copy(query.aliases)
        alias_to_relations = {a: [a] for a in rels}
        joined_attrs = copy(query.joined_attrs)
        relations_to_alias = {}

        partial_select = "SELECT * "

        partial_from, alias = self.recursive_construct(
            partial_order,
            joined_attrs,
            relations_to_alias,
            alias_to_relations,
            aliases,
        )

        partial_where = utils.get_where_clause(query.ast, relations_to_alias, alias, rels)

        partial_query = partial_select + "FROM " + partial_from + " " + partial_where + ";"

        self.counter = 0

        return partial_query


    def construct_partial_query_pg_hint_plan(self, query: Query_Info, partial_order):
        rels_str = str(partial_order)
        rels_str = rels_str.replace('[', '').replace(']', '').replace(' ', '').replace('\'', '')
        rels = rels_str.split(',')

        partial_select = "SELECT * "
        partial_from = self.get_partial_from(query, rels)
        predicates = self.get_partial_selection_predicates(query, rels) + \
                        self.get_partial_join_predicates(query, rels)
        partial_where = "WHERE " + " AND ".join(predicates) if len(predicates) > 0 else ""
        partial_query = partial_select + partial_from + " " + partial_where

        jo_string = self.jo_to_string(partial_order)
        prefix = f'/*+ Leading({jo_string}) */\n'

        if query.generate_queries and len(rels) < query.num_relations:
            query_path = f"{query.query_path}/generated/"
            query_filename = f"{query.filename[:-4]}_{'_'.join(sorted(rels))}.sql"
            cardinality_file = f"{query_path}/cardinalities/{query_filename[:-4]}.csv"
            os.makedirs(query_path, exist_ok=True)
            if not os.path.exists(query_path + query_filename):
                with open(query_path + query_filename, 'w') as f:
                    f.write(partial_query + ";")
                fu.create_cardinality_file(cardinality_file)
                for rel in rels:
                    filtered_cardinality, unfiltered_cardinality = query.cardinalities_full[rel]
                    fu.store_card_for_alias(rel, unfiltered_cardinality, filtered_cardinality, cardinality_file)

        return prefix + partial_query + ";"

    def construct_query(self, query: Query_Info, join_ordering):
        if self.join_setting_tool == jst.PG_HINT_PLAN:
            return self.construct_query_pg_hint_plan(query, join_ordering)
        else:
            return self.construct_query_on_jo(query, join_ordering)

    def construct_query_pg_hint_plan(self, query: Query_Info, join_ordering):
        jo_string = self.jo_to_string(join_ordering)
        prefix = f'/*+ Leading({jo_string}) */\n'
        return prefix + query.sql

    def construct_query_on_jo(self, query: Query_Info, join_ordering):

        aliases = copy(query.aliases)
        alias_to_relations = copy(query.alias_to_relations)
        joined_attrs = copy(query.joined_attrs)
        relations_to_alias = {}

        subq, alias = self.recursive_construct(
            join_ordering,
            joined_attrs,
            relations_to_alias,
            alias_to_relations,
            aliases,
        )

        select_clause = utils.get_select_clause(query.ast, relations_to_alias, alias)
        where_clause = utils.get_where_clause(query.ast, relations_to_alias, alias)

        limit = ""
        if "limit" in query.ast:
            limit = " LIMIT " + str(query.ast["limit"])

        query_str = select_clause + " FROM " + subq + where_clause + limit + ";"

        self.counter = 0

        return query_str

    def recursive_construct(
        self,
        subtree,
        joined_attrs,
        relations_to_alias,
        alias_to_relations,
        aliases,
    ):

        if isinstance(subtree, str):
            return subtree, subtree

        left, left_alias = self.recursive_construct(
            subtree[0],
            joined_attrs,
            relations_to_alias,
            alias_to_relations,
            aliases,
        )
        right, right_alias = self.recursive_construct(
            subtree[1],
            joined_attrs,
            relations_to_alias,
            alias_to_relations,
            aliases,
        )

        new_alias = "J" + str(self.counter)
        relations_to_alias[left_alias] = new_alias
        relations_to_alias[right_alias] = new_alias
        self.counter += 1

        alias_to_relations[new_alias] = [left_alias, right_alias]

        join_info = joined_attrs[(left_alias, right_alias)]
        attr1, attr2 = join_info.attr1, join_info.attr2

        if left == left_alias:
            left = aliases[left] + " AS " + left
        if right == right_alias:
            right = aliases[right] + " AS " + right

        clause = self.select_clause(
            alias_to_relations, left_alias, right_alias, aliases
        )

        subquery = (
            "( SELECT "
            + clause
            + " FROM "
            + left
            + " JOIN "
            + right
            + " on "
            + left_alias
            + "."
            + attr1
            + " = "
            + right_alias
            + "."
            + attr2
            + ") "
            + new_alias
        )

        self.update_joined_attrs((left_alias, right_alias), new_alias, joined_attrs)

        return subquery, new_alias

    def update_joined_attrs(self, old_pair, new_alias, joined_attrs):
        # Delete the two elements corresponding to the subtrees we joined
        # e.g. [A,B]->[id, id2], [B,A]->[id2, id]
        del joined_attrs[(old_pair[0], old_pair[1])]
        del joined_attrs[(old_pair[1], old_pair[0])]

        # Search for other entries with values from the old pair and update their name
        keys = list(joined_attrs.keys())
        for (t1, t2) in keys:

            (rel1, attr1) = (t1, joined_attrs[(t1, t2)].attr1)
            (rel2, attr2) = (t2, joined_attrs[(t1, t2)].attr2)

            if t1 == old_pair[0] or t1 == old_pair[1]:
                rel1 = new_alias
                attr1 = t1 + "_" + attr1

            if t2 == old_pair[0] or t2 == old_pair[1]:
                rel2 = new_alias
                attr2 = t2 + "_" + attr2

            if t1 != rel1 or t2 != rel2:
                del joined_attrs[(t1, t2)]
                joined_attrs[(rel1, rel2)] = utils.JoinInfo(rel1, rel1, attr1, rel2, rel2, attr2)

    def select_clause(
        self, alias_to_relations, left_alias, right_alias, aliases
    ):

        clause = []

        self.recursive_select_clause(
            clause, "", alias_to_relations, left_alias, left_alias, aliases
        )
        self.recursive_select_clause(
            clause, "", alias_to_relations, right_alias, right_alias, aliases
        )

        select_clause = ""
        for i in range(len(clause) - 1):
            select_clause += clause[i] + ", "
        select_clause += clause[len(clause) - 1]

        return select_clause

    def recursive_select_clause(
        self, clause, path, alias_to_relations, alias, base_alias, aliases
    ):

        # print(alias)
        rels = alias_to_relations[alias]
        if len(rels) > 1:
            for rel in rels:
                path1 = path + rel + "_"
                self.recursive_select_clause(
                    clause, path1, alias_to_relations, rel, base_alias, aliases
                )
        else:
            attributes = self.attrs[rels[0]]
            for attr in attributes:
                tmp = path + attr.attr_name
                c = base_alias + "." + tmp + " AS " + base_alias + "_" + tmp
                if c not in clause:
                    clause.append(c)

    def jo_to_string(self, join_order):
        return str(join_order).replace('[', '('). \
                               replace(']', ')'). \
                               replace('\'', ''). \
                               replace(',', '')

    def get_partial_from(self, query: Query_Info, rels):
        from_str = "FROM "
        froms = []

        for rel in rels:
            froms.append(f"{query.database.relations_tables[rel]} AS {rel}")

        from_str += ", ".join(froms)
        return from_str

    def get_partial_selection_predicates(self, query: Query_Info, rels):
        selects = []

        for attr_name, predicates in query.attr_vals.items():
            alias = attr_name.split('.')[0]
            if alias in rels:
                selects.extend(predicates)

        return selects

    def get_partial_join_predicates(self, query: Query_Info, rels):
        join_preds = []

        for join_rels, join_attrs in query.joined_attrs.items():
            if join_rels[0] in rels and join_rels[1] in rels:
                join_preds.append(
                        f"{join_rels[0]}.{join_attrs.attr1} = " \
                        f"{join_rels[1]}.{join_attrs.attr2}")

        return join_preds
