#!/bin/bash

export RLJO_JOB_BASE=$(git rev-parse --show-toplevel)/JOB
export RLJO_PSQL_SRC_DIRECTORY=$(git rev-parse --show-toplevel)/postgres/install

mkdir $RLJO_JOB_BASE
git clone -n https://github.com/danolivo/jo-bench $RLJO_JOB_BASE/jo-bench

cd $RLJO_JOB_BASE/jo-bench
git checkout a2019f9

$RLJO_PSQL_SRC_DIRECTORY/bin/createdb imdbload --port 5432
$RLJO_PSQL_SRC_DIRECTORY/bin/psql -d imdbload -f $RLJO_JOB_BASE/jo-bench/schema.sql --port 5432
$RLJO_PSQL_SRC_DIRECTORY/createuser --superuser postgres --port 5432

sed -i "s/, ENCODING ''WIN1251''//g" $RLJO_JOB_BASE/jo-bench/copy.sql
$RLJO_PSQL_SRC_DIRECTORY/bin/psql -d imdbload -vdatadir="'$RLJO_JOB_BASE/jo-bench'" -f $RLJO_JOB_BASE/jo-bench/copy.sql --port 5432
$RLJO_PSQL_SRC_DIRECTORY/bin/psql -d imdbload -f $RLJO_JOB_BASE/jo-bench/fkindexes.sql --port 5432

cd ..
