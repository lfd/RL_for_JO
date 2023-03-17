#!/bin/bash
#
export RLJO_PSQL_BASE=$(git rev-parse --show-toplevel)/postgres

# PostgreSQL
export RLJO_PSQL_DATA_DIRECTORY=$RLJO_PSQL_BASE/database
export RLJO_PSQL_SRC_DIRECTORY="$RLJO_PSQL_BASE/postgresql-16.0"
export RLJO_PSQL_INSTALL_DIRECTORY="$RLJO_PSQL_BASE/install"
export RLJO_PG_HINT_PLAN_BASE="$RLJO_PSQL_BASE/pg_hint_plan"
export RLJO_PG_HINT_PLAN_SRC_DIRECTORY="$RLJO_PG_HINT_PLAN_BASE/pg_hint_plan-REL16_1_6_0"

if (( $(id -u) == 0 )); then
    echo "Please do not run as root"
    exit
fi

if [ ! -d "$RLJO_PSQL_BASE" ]; then
    mkdir $RLJO_PSQL_BASE
fi

# The following to libraries may be missing
# sudo apt install libreadline-dev
# sudo apt install zlib1g-dev

if [ ! -d "$RLJO_PSQL_INSTALL_DIRECTORY" ]; then
    echo "Installing postgresql"
    cd $RLJO_PSQL_BASE
    wget https://ftp.postgresql.org/pub/source/v16.0/postgresql-16.0.tar.gz
    tar xvfz postgresql-16.0.tar.gz
    mkdir $RLJO_PSQL_INSTALL_DIRECTORY
    mkdir $RLJO_PSQL_DATA_DIRECTORY
    cd $RLJO_PSQL_SRC_DIRECTORY
    ./configure --prefix=$RLJO_PSQL_INSTALL_DIRECTORY --enable-debug
    make -j $(nproc)
    make install
    cd $RLJO_PSQL_BASE
    cd ..
fi

export PATH=$RLJO_PSQL_BASE/install/bin:$PATH
export LD_LIBRARY_PATH=$RLJO_PSQL_BASE/install/lib/:$LD_LIBRARY_PATH

initdb -D $RLJO_PSQL_DATA_DIRECTORY --user=postgres
pg_ctl -D $RLJO_PSQL_DATA_DIRECTORY -l $RLJO_PSQL_BASE/logfile start

## pg_hint_plan
if [ ! -d "$RLJO_PG_HINT_PLAN_BASE" ]; then
    mkdir $RLJO_PG_HINT_PLAN_BASE
    cd $RLJO_PG_HINT_PLAN_BASE
    wget https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/REL16_1_6_0.tar.gz
    tar xvfz REL16_1_6_0.tar.gz
    cd $RLJO_PG_HINT_PLAN_SRC_DIRECTORY
    make -j $(nproc)
    make install
fi

cd $RLJO_PSQL_BASE
cd ..
