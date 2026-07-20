import numpy as np

from amt.configuration import Config
from amt.sel_runner import Experiment

if __name__ == "__main__":
    for n in [20]:
        for mode in ["beta.med"]:
            for m in [1, 2, 4, 6, 8, 10, 15]:
                conf = Config(
                    n = n,
                    m = m,
                    sample_size = 2000,
                    initial_size = 10,
                    reps = 2500,
                    common_p = 0.5,
                    p_diff = 0.05,
                    selection_mode = mode,
                )


                exp = Experiment(conf=conf, save_path = f"./exp_results/coin_res/vari_m")
                exp.run_parallel()
                exp.save()
