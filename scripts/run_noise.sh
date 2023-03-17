#!/bin/bash

# in case the script is not started from within qce24_repro directory
if [ ! "${PWD}" = "/home/repro/qce24_repro" ]; then
    cd /home/repro/qce24_repro/
fi

cd experimental_analysis/

echo "started running noisy evaluation..."

for p in 0.0 0.01 0.02 0.03 0.04 0.05; do
	echo "..started noisy evaluation for quantum actor and depolarising probability $p..."
	source noisy_simulation/call_q_agent.sh $p
	echo "..noisy evaluatin for quantum actor depolarising probability $p done."

	echo "..started noisy evaluation for fully quantum model and depolarising probability $p..."
	source noisy_simulation/call_q_base.sh $p
	echo "..noisy evaluatin for fully quantum model and depolarising probability $p done."
done

# Run training with all frameworks
echo "all noisy evaluations done."

cd ..
