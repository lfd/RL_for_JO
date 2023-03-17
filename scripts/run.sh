#!/bin/bash

if [ $# -eq 0 ]; then
	echo "Usage: ./scripts/run.sh [all|experiments_classic|experiments_quantum|experiment_noise|plot|paper|plot_paper|bash]"
	exit 1
fi

# in case the script is not started from within qce24_repro directory
if [ ! "${PWD}" = "/home/repro/qce24_repro" ]; then
    cd /home/repro/qce24_repro/
fi

export PYTHONPATH=/home/repro/qce24_repro:$PYTHONPATH

cd scripts/

if [[ "$1" = "all" || "$1" = "experiments" || "$1" = "experiments_classic" ||
	"$1" = "experiments_quantum" || "$1" = "experiments_noise" ]]; then
	echo "Setting up JOB first.."
	./setup_JOB_without_cloning.sh
fi

if [ "$1" = "all" ]; then
	./run_classic.sh
	./run_quantum.sh
	./run_noise.sh
	./generate_plots.sh
	./generate_paper.sh
elif [ "$1" = "experiments_classic" ]; then
	./run_classic.sh
elif [ "$1" = "experiments_quantum" ]; then
	./run_quantum.sh
elif [ "$1" = "experiments_noise" ]; then
	./run_noise.sh
elif [ "$1" = "experiments" ]; then
	./run_classic.sh
	./run_quantum.sh
	./run_noise.sh
elif [ "$1" = "plot" ]; then
	./generate_plots.sh
elif [ "$1" = "paper" ]; then
	./generate_paper.sh
elif [ "$1" = "plot_paper" ]; then
	./generate_plots.sh
	./generate_paper.sh
elif [ "$1" = "bash" ]; then
	# launch shell
	cd ..
	/bin/bash
	exit 0
else
    echo "Usage: ./scripts/run.sh [all|experiments_classic|experiments_quantum|experiment_noise|plot|paper|plot_paper|bash]"
fi

cd ..

# launch shell
/bin/bash
