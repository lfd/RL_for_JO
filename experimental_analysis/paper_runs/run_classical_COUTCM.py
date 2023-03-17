import subprocess

# Classical runs for the OUT cost model

## General Hyperparameter
selpreds = 0
mini_batchsize = 32

## Classical - Baseline
take_best_frequency = 20000
take_best_threshold = 0
lr_start = 5e-5
lr_duration = 0.9
batchsize = 20
multistep = 0

for k in range(10):
    p = subprocess.Popen(["python", "hyper_search/hyper_classical_xval.py", "hyper.classical.COUTCM.basic_rejoin_shift", str(lr_start), str(multistep), str(selpreds), str(lr_duration), str(take_best_frequency), str(batchsize), str(mini_batchsize), str(take_best_threshold), str(k)])
p.communicate()


## Classical - Baseline
take_best_frequency = 1
take_best_threshold = 0.1
lr_start = 3e-4
lr_duration = 0.9
batchsize = 20
multistep = 0

for k in range(10):
    p = subprocess.Popen(["python", "hyper_search/hyper_classical_xval.py", "hyper.classical.COUTCM.mod_reduced_rejoin_384", str(lr_start), str(multistep), str(selpreds), str(lr_duration), str(take_best_frequency), str(batchsize), str(mini_batchsize), str(take_best_threshold), str(k)])
p.communicate()
