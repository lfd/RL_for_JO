import random
import copy
import os
from collections import defaultdict
import sys

sys.path.insert(0, os.getcwd())

from src.database.database import Database
from src.database.sql_info import split_dataset, get_queries_incremental
from src.database.database_utils import is_number

def generate_queries(database: Database, num_relations=None, selection_predicates = None, only_num_rels = False, num_queries = 100000, path="generated"):
    if num_relations is None:
        num_relations = random.randint(4, 39)

    if only_num_rels:
        aliases, all_join_attrs, relations_tables = get_relations_and_join_attrs_from_queries(database, num_relations)
    else:
        aliases, all_join_attrs, relations_tables = get_relations_and_join_attrs_from_queries(database, num_relations = None)

    for i in range(num_queries):
        join_attrs, joined_aliases = get_random_join_attributes(database, aliases, all_join_attrs, num_relations)

        from_clause, relations = get_from_clause(database, joined_aliases, relations_tables)
        where_clause = get_where_clause(database, join_attrs, relations, selection_predicates)

        select_clause = "SELECT *"
        query = select_clause + from_clause + where_clause + ";"

        with open(path + f"generated{i}.sql", 'w') as file:
            file.write(query)
        print(f"Generated {path}/generated{i}.sql")

def get_random_join_attributes(database: Database, aliases: dict, all_join_attrs: dict, num_relations: int):
    current_alias = random.choice(list(aliases.keys()))
    joinable_aliases = copy.copy(aliases[current_alias])
    joined_aliases = [current_alias]
    join_attrs = dict()

    while len(joined_aliases) < num_relations:
        current_alias = random.choice(joinable_aliases)
        possible_join_aliases = [j for j in aliases[current_alias] if j in joined_aliases]
        join_alias = random.choice(possible_join_aliases)
        join_attrs[(join_alias, current_alias)] = random.choice(all_join_attrs[(join_alias, current_alias)])
        joined_aliases.append(current_alias)
        joinable_aliases.remove(current_alias)
        joinable_aliases.extend([a for a in copy.copy(aliases[current_alias]) if a not in joined_aliases and a not in joinable_aliases])

    return join_attrs, joined_aliases

def get_from_clause(database: Database, aliases: list, relations_tables: dict):
    from_clause = "\nFROM "
    relations = { alias: relation for alias, relation in relations_tables.items() if alias in aliases }
    relation_str = ", ".join([f'{rel} {alias}' for alias, rel in relations.items()])
    return from_clause + relation_str, relations

def get_where_clause(database: Database, join_attrs: dict, relations: dict, selection_predicates = None):
    where_clause = "\nWHERE "
    sel_preds_query = defaultdict(list)
    sel_preds_str = ""

    if selection_predicates is None:
        selection_predicate_str = ""
    else:
        num_preds = 0
        for alias, relation in relations.items():
            if selection_predicates[relation]:
                for attr, most_common_vals in selection_predicates[relation]:
                    sel_preds_query[f'{alias}.{attr}'] = most_common_vals
                num_preds += len(selection_predicates[relation])

        max_num_predicates = random.randint(0, num_preds//4)

        sel_attrs = random.choices(list(sel_preds_query.keys()), k = max_num_predicates)
        for sel_attr in sel_attrs:
            sel_pred = random.choice(sel_preds_query[sel_attr])

            if is_number(sel_pred):
                sign = random.choice(['=', '>', '<'])
            else:
                sign = '='
                sel_pred = f"{sel_pred}"

            sel_preds_str += f'{sel_attr} {sign} {sel_pred} AND '

    join_predicates = [f'{alias1}.{join_attrs[(alias1,alias2)].attr1}={alias2}.{join_attrs[(alias1,alias2)].attr2}' \
            for (alias1, alias2) in join_attrs]
    join_predicate_str = " AND ".join(join_predicates)
    return where_clause + sel_preds_str + join_predicate_str

def get_selection_predicates(database: Database):
    q = "SELECT s.tablename, s.attname, c.data_type, s.most_common_vals " \
        "FROM pg_stats s, information_schema.columns c " \
        "WHERE s.tablename = c.table_name AND c.column_name = s.attname AND " \
        "s.schemaname = 'public' AND s.most_common_vals is not NULL;"

    cursor = database.conn.cursor()
    cursor.execute(q)
    rows = cursor.fetchall()
    cursor.close()

    attrs = defaultdict(list)

    for tablename, attname, datatype, most_common_vals in rows:
        vals = []
        if attname[-3:] == "_id":
            continue
        if most_common_vals[0] == '{':
            most_common_vals = most_common_vals[1:-1].split(',')
            v = ""
            for val in most_common_vals:
                if "\\" in val:
                    val = val.replace("\\", "")
                if "\'" in val:
                    continue

                if val[0] == "\"":
                    if val[-1] == "\"":
                        val = val[1:-1]
                        if datatype == 'integer':
                            vals.append(val)
                        else:
                            vals.append(f'\'{val}\'')
                    else:
                        v = val[1:] + ","
                elif val[-1] == "\"":
                    v += val[:-1]
                    if datatype == 'integer':
                        vals.append(v)
                    else:
                        vals.append(f'\'{v}\'')
                    v = ""
                else:
                    if len(v) == 0:
                        if datatype == 'integer':
                            vals.append(val)
                        else:
                            vals.append(f'\'{val}\'')
                    else:
                        v += val

        attrs[tablename].append((attname, vals))

    return attrs

def get_relations_and_join_attrs_from_queries(database, num_relations):
    all_queries, _ = split_dataset(database.base_query_path, database, num_relations, gather_sel_info=False)
    queries = get_queries_incremental(all_queries)

    join_attrs = defaultdict(list)
    relations_tables = dict()
    aliases = defaultdict(list)

    for q in queries:
        for attrs_key, attrs_val in q.joined_attrs.items():
            if not attrs_val in join_attrs[attrs_key]:
                join_attrs[attrs_key].append(attrs_val)
        for alias, table in q.aliases.items():
            relations_tables[alias] = table
        for (alias, join_alias) in join_attrs:
            if join_alias not in aliases[alias]:
                aliases[alias].append(join_alias)
            if alias not in aliases[join_alias]:
                aliases[join_alias].append(alias)

    return aliases, join_attrs, relations_tables

if __name__ == '__main__':
    num_relations = int(sys.argv[1]) if len(sys.argv) >= 2 else None
    only_num_rels = bool(int(sys.argv[2])) if len(sys.argv) >= 3 else False
    use_selection_predicates = bool(int(sys.argv[3])) if len(sys.argv) >= 4 else True

    db = Database(collect_db_info=True, base_query_path="queries/jo-bench/")
    path = f'queries/generated/v1/rels{num_relations}/'
    os.makedirs(path, exist_ok=True)

    selection_predicates = get_selection_predicates(db) if use_selection_predicates else None

    q = generate_queries(db, num_relations, selection_predicates, only_num_rels, path=path)

