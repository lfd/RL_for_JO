import subprocess

# Classical runs for PostgreSQL-V8

## General Hyperparameter
selpreds = 0
mini_batchsize = 32

## Classical - Baseline
take_best_frequency = 20000
take_best_threshold = 0
lr_start = 9e-5
lr_duration = 0.9
batchsize = 20
multistep = 0

for k in range(10):
    p = subprocess.Popen(["python", "hyper_search/hyper_classical_xval.py", "hyper.classical.PGCM8.basic_rejoin_shift", str(lr_start), str(multistep), str(selpreds), str(lr_duration), str(take_best_frequency), str(batchsize), str(mini_batchsize), str(take_best_threshold), str(k)])
p.communicate()

