'''
Build database related info
- tables,
- relations (original-tables + aliases),
- {relation : attributes}
- attributes

Original implementation by:
Guo Xintong
https://github.com/GUOXINTONG/rejoin
Commit: 02365ab0f1cb2e0019be062babf0d66aaa93c4eb
File: src/database.py

Modified.
'''

import configs.database as creds
import psycopg2
from psycopg2.errors import QueryCanceled
import random
import moz_sql_parser
from collections import defaultdict
import json
import os
from threading import Thread
from math import inf

from src.database.query_parser import Query_Parser
from src.database.database_utils import JO_Setting_Tool as jst, get_joo_server, is_number, Attribute
from src.database.file_utils import store_db_info, load_db_info

class Database:

    def __init__(self, collect_db_info,
                 jo_setting_tool = jst.NATIVE,
                 base_query_path = "queries/jo-bench/",
                 timeout=300,
                 use_geqo=False):
        self.jo_setting_tool = jo_setting_tool
        self.joo_server = get_joo_server(jo_setting_tool)

        self.conn = self.connect()
        self.aliases = {}

        self.base_query_path = base_query_path

        self.db_info_file = f"{base_query_path}/db_info.csv"

        if collect_db_info:
            self.tables, self.relations, \
            self.relations_attributes, \
            self.relations_tables, \
            self.all_join_attrs, \
            self.attributes, \
            self.relations_cardinalities = (
                self.get_relations_attributes()
            )

            self.query_parser = Query_Parser(jo_setting_tool, self.relations_attributes)

        self.timeout = timeout
        self.use_geqo = use_geqo

    def connect(self):
        # local postgres connection, need to configure in config/database.py
        try:
            conn_string = (
                "host="
                + creds.PGHOST
                + " port="
                + creds.PGPORT
                + " dbname="
                + creds.PGDATABASE
                + " user="
                + creds.PGUSER
                + " password="
                + creds.PGPASSWORD
            )
            conn = psycopg2.connect(conn_string)

            if self.jo_setting_tool == jst.PG_HINT_PLAN:
                conn.cursor().execute("LOAD \'pg_hint_plan\';")

            return conn
        except (Exception, psycopg2.Error) as error:
            exit(f"Failed to connect to database\n{error}")

    def get_relations_attributes(self):
        """
        Returns relations and their attributes

        Uses tables/attributes from the database but also aliases found on the dataset's queries

        Args:
            None
        Returns:
            relations: list ['alias1','alias2', ..]
            relations_attributes: dict {'alias1':['attr1','attr2', ..], ..}
            relations_tables: dict {'alias1':'table1', ..}
            all_join_attrs: dict {('alias1','alias2'):('join_attr1','join_attr2'), ..}

        """
        db_info = load_db_info(self.db_info_file)
        cursor = None

        if len(db_info) == 0:
            cursor = self.conn.cursor()
            q = (
                "SELECT c.table_name, c.column_name FROM information_schema.columns c "
                "INNER JOIN information_schema.tables t ON c.table_name = t.table_name "
                "AND c.table_schema = t.table_schema "
                "AND t.table_type = 'BASE TABLE' "
                "AND t.table_schema = 'public' "
            )
            cursor.execute(q)
            db_info = cursor.fetchall()

        tables_attributes = {}
        tables_cardinalities = {}

        for i in range(len(db_info)):
            table = db_info[i][0]
            attribute_name = db_info[i][1]
            if len(db_info[i]) == 2:
                q = "SELECT s.n_distinct, c.data_type, r.reltuples " \
                    "FROM pg_stats s, pg_class r, information_schema.columns c " \
                    "WHERE s.tablename = c.table_name AND c.column_name = s.attname AND " \
                    f"r.relname = s.tablename AND " \
                    f"s.schemaname = 'public' AND s.tablename = '{table}' AND " \
                    f"s.attname = '{attribute_name}'"
                cursor.execute(q)

                n_distinct, data_type, cardinality = cursor.fetchone()

                if data_type in ["integer", "smallint", "bigint", "real"]:
                    q = f"SELECT min({attribute_name}), max({attribute_name}) FROM {table} "
                    cursor.execute(q)
                    min, max = cursor.fetchone()
                else:
                    min, max = None, None
                store_db_info(table, attribute_name, n_distinct, data_type, cardinality, min, max, self.db_info_file)
            else:
                n_distinct, data_type, cardinality, min, max = db_info[i][2:]

            attribute = Attribute(table, attribute_name, n_distinct, cardinality, min, max,
                                  is_number = data_type in ["integer", "smallint", "bigint", "real"])

            if table in tables_attributes:
                tables_attributes[table].append(attribute)
            else:
                tables_attributes[table] = [attribute]

            tables_cardinalities[table] = cardinality

        if cursor is not None:
            cursor.close()

        tables = list(tables_attributes.keys())
        relations_attributes = {}
        relations = []
        relations_tables = {}
        relations_cardinalities = {}
        all_join_attrs = defaultdict(list)
        attributes = dict()

        queries = os.listdir(self.base_query_path)

        for q in queries:
            if not q[-4:] == ".sql":
                continue

            with open(self.base_query_path + q, 'r') as query_file:
                ast = moz_sql_parser.parse(query_file.read())

            for r in ast["from"]:
                if r["name"] not in relations:
                    relations.append(r["name"])
                    relations_attributes[r["name"]] = tables_attributes[r["value"]]
                    relations_cardinalities[r["name"]] = tables_cardinalities[r["value"]]
                    relations_tables[r["name"]] = r["value"]
                    for attr in tables_attributes[r["value"]]:
                        attributes[f'{r["name"]}.{attr.attr_name}'] = attr

            for v in ast["where"]["and"]:
                if (
                    "eq" in v
                    and isinstance(v["eq"][0], str)
                    and isinstance(v["eq"][1], str)
                ):
                    table_left = v["eq"][0].split(".")[0]
                    table_right = v["eq"][1].split(".")[0]
                    attr1 = attributes[v["eq"][0]]
                    attr2 = attributes[v["eq"][1]]

                    all_join_attrs[(table_left, table_right)].append((attr1, attr2))
                    all_join_attrs[(table_right, table_left)].append((attr2, attr1))

        return tables, relations, relations_attributes, relations_tables, all_join_attrs, attributes, relations_cardinalities

    def close(self):
        if self.conn:
            self.conn.close()

    def get_query_plan(self, query, jst_type = jst.NATIVE, estimation = True, high_level = False):
        if estimation:
            query = "EXPLAIN (FORMAT JSON) " + query + ";"
        else:
            query = "EXPLAIN (ANALYZE TRUE, FORMAT JSON) " + query + ";"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)

            if jst_type == jst.QC4DB_PLUGIN:
                server_thread.join()

            rows = cursor.fetchone()
            plan = rows[0][0]
            if not high_level:
                plan = plan["Plan"]
        except QueryCanceled:
            print("timeout")
            plan = None
            self.conn.rollback()
        cursor.close()
        return plan

    def optimizer_cost(self, query, force_order = False, jst_type = jst.NATIVE, use_geqo = False):

        if jst_type == jst.QC4DB_PLUGIN:
            server_thread = Thread(target = self.joo_server.run, args=(join_order,))
            server_thread.start()

        self.execute_prefix(force_order, jst_type, use_geqo=use_geqo)

        if jst_type == jst.NATIVE:
            cursor = self.conn.cursor()
            query = "EXPLAIN " + query
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            row = rows[0][0]
            cost = float(row.split("cost=")[1].split("..")[1].split(" ")[0])
        else:
            plan = self.get_query_plan(query, jst_type)
            cost = plan["Total Cost"]

        return cost

    def optimizer_card(self, query):
        self.execute_prefix(False, self.jo_setting_tool)

        plan = self.get_query_plan(query, self.jo_setting_tool)

        card = plan["Plan Rows"]

        return card

    def get_query_time(self, query, force_order = False, jst_type = jst.NATIVE, timeout = None):

        if jst_type == jst.QC4DB_PLUGIN:
            server_thread = Thread(target = self.joo_server.run, args=(join_order,))
            server_thread.start()

        self.execute_prefix(force_order, jst_type, timeout)
        plan = self.get_query_plan(query, self.jo_setting_tool, estimation=False, high_level=True)

        if plan is not None:
            execution = plan["Execution Time"]
            planning = plan["Planning Time"]
        else:
            execution = self.timeout * 1000
            planning = None

        if jst_type == jst.QC4DB_PLUGIN:
            server_thread.join()

        return planning, execution

    def execute_prefix(self, force_order = False, jst_type = jst.NATIVE, timeout = None, use_geqo = False):

        prefix = ""

        if jst_type == jst.PG_HINT_PLAN:
            prefix += "SET pg_hint_plan.enable_hint = "
            prefix += "on;\n" if force_order else "off;\n"
        elif jst_type == jst.NATIVE:
            prefix += "SET join_collapse_limit = "
            prefix += "1;\n" if force_order else "20;\n"

        prefix += "SET statement_timeout = "
        prefix += f'\'{self.timeout}s\';\n' if timeout is None else f'\'{timeout}ms\';\n'

        prefix += "SET geqo = "
        prefix += "on;\n" if (self.use_geqo or use_geqo) and not force_order else "off;\n"

        prefix += "SET geqo_threshold = "
        prefix += "2;\n" if (self.use_geqo or use_geqo) and not force_order else "200;\n"

        cursor = self.conn.cursor()
        cursor.execute(prefix)
        cursor.close()

    def get_reward(self, query, cost_based, force_order, jst_type = jst.NATIVE):
        if cost_based:
            return self.optimizer_cost(query, force_order, jst_type)  # Get Cost Model's Estimate
        return self.get_query_time(query, force_order, jst_type)[1]  # Get actual query-execution latency

    def get_num_tuples(self, table, alias, predicates, estimation=False):
        cursor = self.conn.cursor()
        where_clause = "" if len(predicates) == 0 else f"WHERE {' AND '.join(predicates)}"
        q = f"SELECT * FROM {table} AS {alias} {where_clause}"
        return self.get_count_result(q, self.jo_setting_tool, estimation)

    def get_num_tuples_for_join(self, table1, table2, join_attr1, join_attr2):
        attr1 = f"{table1}.{join_attr1}"
        attr2 = f"{table2}.{join_attr2}"

        cursor = self.conn.cursor()
        q = f"SELECT COUNT(*) FROM {table1}, {table2} WHERE {attr1} = {attr2}"
        cursor.execute(q)
        num_tuples = cursor.fetchone()[0]
        cursor.close()

        return num_tuples

    def get_count_result(self, query, jst_type=jst.NATIVE, estimation=False):
        self.execute_prefix(False, jst_type)
        plan = self.get_query_plan(query, jst_type, estimation)
        if plan is None:
            return None
        if estimation:
            return int(plan['Plan Rows'])
        else:
            return int(plan['Actual Rows'])

