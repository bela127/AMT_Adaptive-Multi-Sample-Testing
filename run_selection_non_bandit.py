import numpy as np

from amt.configuration import Config
from amt.sel_runner import Experiment

if __name__ == "__main__":
    for n in [20]:
        for mode in ["equal", "ts", "ts.5", "means", "mean.slow", "beta.med", "beta.max", "evar"]:# "beta.med", "beta.max", "evar"
            for m in [1]:
                for p in [0.5]:
                    conf = Config(
                        n = n,
                        m = m,
                        sample_size = 2000,
                        initial_size = 10,
                        reps = 2500,
                        common_p = p,
                        p_diff = 0.1,
                        selection_mode = mode,
                    )


                    exp = Experiment(conf=conf, save_path = f"./exp_results/coin_res/non_bandit_selection")
                    exp.run_parallel()
                    exp.save()
