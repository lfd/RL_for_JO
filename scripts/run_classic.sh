#!/bin/bash

# in case the script is not started from within qce24_repro directory
if [ ! "${PWD}" = "/home/repro/qce24_repro" ]; then
    cd /home/repro/qce24_repro/
fi

cd experimental_analysis/

echo "started running classical RL trainings..."

# Part of supplementary material study
## To run this PostgreSQL-V8 instead of PostgreSQL-V16 needs to be installed
## python paper_runs/run_classical_PGCM8.py
python paper_runs/run_classical_PGCM16.py
python paper_runs/run_classical_COUTCM.py

# Part of main paper study
python paper_runs/run_classical_rel4.py

# Run training with all frameworks
echo "all classical RL trainings done."

cd ..
