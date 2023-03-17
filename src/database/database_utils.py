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
File: src/database_utils.py

Modified.
'''

from enum import Enum, auto
import moz_sql_parser

from src.database.joo_server import JooServer

def get_operator_map():
    operator_map = {
        "eq": "=",
        "neq": "!=",
        "gt": ">",
        "lt": "<",
        "gte": ">=",
        "lte": "<=",
        "like": "LIKE",
        "not_like": "NOT LIKE",
        "in": "IN",
        "between": "BETWEEN",
        "missing": "IS NULL",
        "exists": "IS NOT NULL"
    }
    return operator_map

def get_alias(attr, relations_to_alias, alias, rels = None):

    tmp = attr.split(".")
    relname = tmp[0]
    attrname = tmp[1]

    if rels is not None and relname not in rels:
        return None

    while relname in relations_to_alias:
        attrname = relname + "_" + attrname
        relname = relations_to_alias[relname]

    return alias + "." + attrname


def get_select_clause(query_ast, relations_to_alias, alias):

    select_operator_map = {
        "min": "MIN",
        "max": "MAX",
    }  # to be filled with other possible values

    select_stmt = query_ast["select"]
    select = []

    if not isinstance(select_stmt, (list,)):
        select_stmt = [select_stmt]

    for v in select_stmt:

        if v == "*":
            return "SELECT *"

        val = v["value"]
        name = ""

        if not isinstance(val, str):
            key = list(val.keys())[0]
            val = (
                select_operator_map[key]
                + "("
                + get_alias(val[key], relations_to_alias, alias)
                + ")"
            )
        else:
            val = get_alias(val, relations_to_alias, alias)

        if "name" in v:
            name = " AS " + v["name"]

        select.append((val, name))

    select_clause = "SELECT "
    for i in range(len(select) - 1):
        select_clause += select[i][0] + select[i][1] + ", \n"
    select_clause += select[len(select) - 1][0] + select[len(select) - 1][1]

    return select_clause


def construct_stmt(stmt, relations_to_alias, alias, rels = None):

    key = list(stmt.keys())[0]
    operator_map = get_operator_map()

    if key == "and" or key == "or":  # Need to go deeper

        s = where_and_or(stmt, relations_to_alias, alias, rels)
        if s == "":
            return s
        else:
            return "( " + s + " )"
    else:
        if key == "between":

            if isinstance(stmt[key][1], dict):
                left = "'" + stmt[key][1]["literal"] + "'"
            else:
                left = str(stmt[key][1])

            if isinstance(stmt[key][2], dict):
                right = "'" + stmt[key][2]["literal"] + "'"
            else:
                right = str(stmt[key][2])

            rvalue = left + " AND " + right

        elif isinstance(stmt[key][1], dict):  # Dict (Naively assuming it's a literal)

            lit = stmt[key][1]["literal"]

            if isinstance(lit, list):
                rvalue = " ( "
                for i in range(len(lit) - 1):
                    rvalue = rvalue + "'" + lit[i] + "', "
                rvalue = rvalue + "'" + lit[len(lit) - 1] + "' ) "

            elif key == "in":
                rvalue = "( '" + lit + "' )"

            else:
                rvalue = "'" + lit + "'"

        elif isinstance(stmt[key][1], int):  # Integer
            rvalue = str(stmt[key][1])

        elif key == 'missing' or key =='exists':
            rvalue  = ""
        else:
            rvalue = get_alias(stmt[key][1], relations_to_alias, alias, rels)
            if rvalue is None:
                return None

        if key == 'missing' or key =='exists':
            alias = get_alias(stmt[key], relations_to_alias, alias, rels)
            if alias is None:
                return None
            query_pass_str = alias + " " + operator_map[key]
        else:
            alias = get_alias(stmt[key][0], relations_to_alias, alias, rels)
            if alias is None:
                return None
            query_pass_str = alias + " " + operator_map[key] + " " + rvalue

        return query_pass_str


def where_and_or(where_ast, relations_to_alias, alias, rels = None):

    where_and_clause = ""
    if "and" in where_ast:
        and_stmt = where_ast["and"]
        where_and = []
        for v in and_stmt:
            if not (
                "eq" in v
                and isinstance(v["eq"][0], str)
                and isinstance(v["eq"][1], str)
            ):  # if not a joining
                s = construct_stmt(v, relations_to_alias, alias, rels)
                if s is not None and s != "":
                    where_and.append(s)

        size = len(where_and)
        if size > 0:
            for i in range(size - 1):
                where_and_clause += where_and[i] + " AND \n"
            where_and_clause += where_and[size - 1]

    where_or_clause = ""
    if "or" in where_ast:
        or_stmt = where_ast["or"]
        where_or = []
        for v in or_stmt:
            if not (
                "eq" in v
                and isinstance(v["eq"][0], str)
                and isinstance(v["eq"][1], str)
            ):  # if not a joining
                s = construct_stmt(v, relations_to_alias, alias, rels)
                if s is not None and s != "":
                    where_or.append(s)

        size = len(where_or)
        if size > 0:
            for i in range(size - 1):
                where_or_clause += where_or[i] + " OR \n"
            where_or_clause += where_or[size - 1]

    return where_and_clause + where_or_clause


def get_where_clause(query_ast, relations_to_alias, alias, rels = None):

    where_ast = query_ast["where"]
    where_clause = where_and_or(where_ast, relations_to_alias, alias, rels)

    if where_clause != "":
        return " \nWHERE \n" + where_clause

    else:
        return ""


class JO_Setting_Tool(Enum):
    NATIVE = auto(),
    PG_HINT_PLAN = auto(),
    QC4DB_PLUGIN = auto(),


def create_query_parser(jst_type):

    if jst_type == JO_Setting_Tool.NATIVE:
        return QueryParser_ON_JO()
    elif jst_type == JO_Setting_Tool.PG_HINT_PLAN:
        return QueryParser_pg_hint_plan()

    return None

def get_joo_server(jst_type):

    if not jst_type == JO_Setting_Tool.QC4DB_PLUGIN:
        return None

    return JooServer()

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_jo_for_plan(plan):
    cond_str = ""

    if 'Plans' in plan and len(plan['Plans']) == 2:
        plans = plan['Plans']

        left = get_jo_for_plan(plans[0])
        right = get_jo_for_plan(plans[1])
        return ([left, right])

    elif 'Alias' in plan:
        return plan['Alias']

    elif 'Plans' in plan and len(plan['Plans']) == 1:
        return get_jo_for_plan(plan['Plans'][0])

    else:
        return None

def flatten_nested_str_list(l):
    s = str(l)
    s = s.replace('[', '').replace(']', '').replace(' ', '').replace('\'', '')
    return s.split(',')


class Attribute:
    def __init__(self, table, attr_name, n_distinct, cardinality, min=None, max=None, is_number = False):
        self.table = table
        self.attr_name = attr_name
        self.min = min
        self.max = max
        self.is_number = is_number
        self.n_distinct = n_distinct
        self.cardinality = cardinality

class JoinInfo:
    def __init__(self, table1, alias1, attr1, table2, alias2, attr2):
        self.table1 = table1
        self.table2 = table2
        self.alias1 = alias1
        self.attr1 = attr1
        self.alias2 = alias2
        self.attr2 = attr2
        self.selectivity = None

    def set_selectivity(self, selectivity):
        self.selectivity = selectivity

def toBitString(all, active):
    result = 0
    result |= 1 << all.index(active)
    return result

def generate_all_join_orders(rels, join_preds):
    '''
    Source: https://github.com/TobiasWinker/QC4DB_QO/blob/main/util/joinHelper.py
    Create a list with all possible join trees

    The list is created using dynamic programming
    Uses bit strings instead of sets to check for overlap between subtrees
    This should be significantly faster
    '''
    n = len(rels)
    # dynamic programming
    level = [[] for _ in range(n + 1)]
    # level 1
    level[1].extend([(e, toBitString(rels, e)) for e in rels])
    # level 2 - n
    for new in range(2, n + 1):
        for i in range(1, (new // 2) + 1):
            for idL in range(len(level[i])):
                for idR in range(len(level[new - i])):
                    if (i == new - i) and (idR <= idL):
                        continue
                    left = level[i][idL]
                    right = level[new - i][idR]
                    if (left[1] & right[1]) == 0 and is_cross_free_join([left[0], right[0]], join_preds):
                        level[new].append(([left[0], right[0]], left[1] | right[1]))

    return [e[0] for e in level[n]]

def remove_cross_joins(all_jo, join_preds):
    jo_no_xjoin = []
    for jo in all_jo:
        if is_cross_free_join(jo, join_preds):
            jo_no_xjoin.append(jo)
    return jo_no_xjoin

def is_cross_free_join(jo, join_preds):
    if isinstance(jo, str):
        return True
    left = jo[0]
    right = jo[1]
    if isinstance(left, str) and isinstance(right, str):
        return (left, right) in join_preds or (right, left) in join_preds
    left_aliases = flatten_nested_str_list(left)
    right_aliases = flatten_nested_str_list(right)
    for left_alias in left_aliases:
        for right_alias in right_aliases:
            if (left_alias, right_alias) in join_preds or \
                    (right_alias, left_alias) in join_preds:
                return is_cross_free_join(left, join_preds) and is_cross_free_join(right, join_preds)
    return False

