import subprocess
import sys

hyper_params = {
        "take_best_frequency": [20000],
        "take_best_threshold": [0],
        "lr_start": [5e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4],
        "lr_duration": [0, 0.5, 0.9],
        "multistep": [0],
        "selpreds": [0],
        "batchsize": [5, 10, 20, 50, 10],
        "mini_batchsize": [32],
}

for lr_start in hyper_params["lr_start"]:
    for multistep in hyper_params["multistep"]:
        for selpreds in hyper_params["selpreds"]:
            for lr_duration in hyper_params["lr_duration"]:
                for take_best_frequency in hyper_params["take_best_frequency"]:
                    for take_best_threshold in hyper_params["take_best_threshold"]:
                        for batch in hyper_params["batchsize"]:
                            for mbatch in hyper_params["mini_batchsize"]:
                                for k in range(10):
                                    p = subprocess.Popen(["python", "hyper_search/hyper_classical_xval.py", sys.argv[1], str(lr_start), str(multistep), str(selpreds), str(lr_duration), str(take_best_frequency), str(batch), str(mbatch), str(take_best_threshold), str(k)])
                                p.communicate()
