'''
Stores queries in CSV file.

Original implementation by:
Guo Xintong
https://github.com/GUOXINTONG/rejoin
Commit: 02365ab0f1cb2e0019be062babf0d66aaa93c4eb
File: queries2db.py

Modified.
'''

import moz_sql_parser
import json
import os
import csv
from collections import defaultdict

sys.path.insert(0, os.getcwd())

from src.database.database import Database

def main():
    db = Database(collect_db_info=False)

    query_dir = "queries/generated/JOB_splits_rels4/split01/test_queries/" # change to specific query dir
    files = os.listdir(query_dir)
    files = [ f for f in files if f[-4:] == '.sql' ]

    cursor = db.conn.cursor()

    query_csv_file = open(f'{query_dir}/data.csv', mode='w')
    query_csv_writer = csv.writer(query_csv_file, delimiter=',')
    query_csv_writer.writerow(['file_name', 'num_relations', 'planning_time', 'execution_time', 'cost_dp', 'cost_geqo'])
    queries = []

    for file_name in files:
        print(file_name)
        with open(f'{query_dir}/{file_name}', "r") as query_file:
            query = query_file.read()
        ast = moz_sql_parser.parse(query)
        queries.append(ast)
        num_relations = len(ast["from"])
        # planning, execution = db.get_query_time(query)
        cost_dp, cost_geqo = db.optimizer_cost(query), db.optimizer_cost(query, use_geqo=True)
        if cost_dp > 0 and num_relations > 3:
            query_csv_writer.writerow([file_name, num_relations, -1, -1, cost_dp, cost_geqo])

    query_csv_file.close()

    cursor.close()

if __name__ == '__main__':
    main()
