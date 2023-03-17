#!/bin/bash

# in case the script is not started from within qce24_repro directory
if [ ! "${PWD}" = "/home/repro/qce24_repro" ]; then
    cd /home/repro/qce24_repro/
fi

cd experimental_analysis/

echo "started running quantum RL trainings..."

# Part of main paper study
for l in 4 8 12 16 20; do
	echo "..started running fully quantum model training with $l layers..."
	python paper_runs/run_fullq_rel4.py $l
	echo "..fully quantum model training with $l layers done."
	echo "..started running quantum actor model training with $l layers..."
	python paper_runs/run_qagent_rel4.py $l
	echo "..quantum actor model training with $l layers done."
	echo "..started running quantum critic model training with $l layers..."
	python paper_runs/run_qcritic_rel4.py $l
	echo "..quantum critic model training with $l layers done."
done

# Run training with all frameworks
echo "all quantum RL trainings done."

cd ..
