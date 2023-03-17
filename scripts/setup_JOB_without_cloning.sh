#!/bin/bash

# This script is not intended to be called on your local system, but as an
# entry point to the docker image

export RLJO_JOB_BASE=/home/repro/JOB
export RLJO_PSQL_SRC_DIRECTORY=/home/repro/postgres/install
export RLJO_PSQL_DATA_DIRECTORY=/home/repro/postgres/database

# start DB instance
pg_ctl -D $RLJO_PSQL_DATA_DIRECTORY -l $RLJO_PSQL_BASE/logfile start
export RLJO_JOB_BASE=/home/repro/JOB
export RLJO_PSQL_SRC_DIRECTORY=/home/repro/postgres/install

# Copy imdbload data to database (this takes time)
cd $RLJO_JOB_BASE/jo-bench
git checkout a2019f9
$RLJO_PSQL_SRC_DIRECTORY/createuser --superuser postgres --user=postgres
$RLJO_PSQL_SRC_DIRECTORY/bin/createdb imdbload --user=postgres
$RLJO_PSQL_SRC_DIRECTORY/bin/psql -d imdbload -f $RLJO_JOB_BASE/jo-bench/schema.sql --user=postgres

sed -i "s/, ENCODING ''WIN1251''//g" $RLJO_JOB_BASE/jo-bench/copy.sql
$RLJO_PSQL_SRC_DIRECTORY/bin/psql -d imdbload -vdatadir="'$RLJO_JOB_BASE/jo-bench'" -f $RLJO_JOB_BASE/jo-bench/copy.sql --user=postgres
$RLJO_PSQL_SRC_DIRECTORY/bin/psql -d imdbload -f $RLJO_JOB_BASE/jo-bench/fkindexes.sql --user=postgres

cd ..
