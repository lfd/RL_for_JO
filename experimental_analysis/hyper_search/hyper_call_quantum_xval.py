import subprocess

hyper_params = {
        "mode": ["base", "agent", "critic", "classical"],
        "take_best_frequency": [1],
        "lr_start": [3e-4, 8e-5],
        "lr_duration": [0.9],
        "multistep": [1],
        "selpreds": [0],
        "data_reuploading": [0, 1],
        "num_layers": [4, 8, 12, 16],
        "nodes": [128],
        "batchsize": [10, 20],
        "mini_batchsize": [32],
        "cost_model": ["COUTCM", "PGCM16"],
        "postprocessing": ["classical", "minimal"],
        "depol_error_prob": [0, 0.01, 0.05, 0.1],
}

for lr_start in hyper_params["lr_start"]:
    for multistep in hyper_params["multistep"]:
        for selpreds in hyper_params["selpreds"]:
            for lr_duration in hyper_params["lr_duration"]:
                for take_best_frequency in hyper_params["take_best_frequency"]:
                    for dr in hyper_params["data_reuploading"]:
                        for nl in hyper_params["num_layers"]:
                            if nl == 4 and dr == 1:
                                continue
                            for mode in hyper_params["mode"]:
                                if mode == "classical" and nl != 4:
                                    continue
                                for batch in hyper_params["batchsize"]:
                                    for mbatch in hyper_params["mini_batchsize"]:
                                        for cm in hyper_params["cost_model"]:
                                            for post_proc in hyper_params["postprocessing"]:
                                                if post_proc == "minimal" and mode == "classical":
                                                    continue
                                                for err_prob in hyper_params["depol_error_prob"]:
                                                    if mode == "classical" and err_prob > 0:
                                                        continue
                                                    for k in range(10):
                                                        if mode == "classical":
                                                            filename = f"hyper_search/hyper_{mode}.py"
                                                        else:
                                                            filename = f"hyper_search/hyper_quantum_{mode}.py"
                                                            if post_proc == "minimal":
                                                                filename = f"hyper_search/hyper_quantum_{mode}_min.py"

                                                        if mode == "agent" or mode == "critic" or mode == "classical":
                                                            for node in hyper_params["nodes"]:
                                                                p = subprocess.Popen(["python", filename, f"hyper.quantum.{cm}.base", str(lr_start), str(multistep), str(selpreds), str(lr_duration), str(take_best_frequency), str(dr), str(nl), str(node), str(batch), str(mbatch), str(k), str(err_prob)])
                                                        else:
                                                            p = subprocess.Popen(["python", filename, f"hyper.quantum.{cm}.base", str(lr_start), str(multistep), str(selpreds), str(lr_duration), str(take_best_frequency), str(dr), str(nl), str(batch), str(mbatch), str(k), str(err_prob)])
                                                    p.communicate()

