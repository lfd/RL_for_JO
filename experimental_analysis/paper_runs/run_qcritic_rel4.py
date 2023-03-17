import subprocess
import sys

# Quantum critic runs for four relations and cost model OUT

## General Hyperparameter
selpreds = 0
mini_batchsize = 32

## Paper Hyperparameter
take_best_frequency = 1
lr_start = 3e-4
lr_duration = 0.9
multistep = 1
nodes = 128
batchsize = 20
err_prob = 0

data_reupl = [0, 1]
if len(sys.argv) < 2:
    nl = 4
else:
    nl = int(sys.argv[1])

for dr in data_reupl:
    if nl == 4 and dr:
        continue
    for k in range(10):
        p = subprocess.Popen(["python", "hyper_search/hyper_quantum_critic.py", f"hyper.quantum.COUTCM.base", str(lr_start), str(multistep), str(selpreds), str(lr_duration), str(take_best_frequency), str(dr), str(nl), str(nodes), str(batchsize), str(mini_batchsize), str(k), str(err_prob)])
    p.communicate()

