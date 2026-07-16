import numpy as np

from amt.configuration import Config
from amt.sel_runner import Experiment

if __name__ == "__main__":
    for n in [20]:#5,10,15,20
        for mode in ["equal", "beta.med"]:# "beta.med", 
            for m in [1]: # 0, 1, 2, int(n/2)
                for hyp in [0, 1]:
                    for p_diff in [0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]:
                        conf = Config(
                            n = n,
                            m = m,
                            sample_size = 2000,
                            initial_size = 10,
                            reps = 2500,
                            common_p = 0.5,
                            p_diff = p_diff,
                            selection_mode = mode,
                            hyp=hyp
                        )


                        exp = Experiment(conf=conf, save_path = f"./exp_results/coin_res")
                        exp.run_parallel()
                        exp.save()
