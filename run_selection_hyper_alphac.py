import numpy as np

from amt.configuration import Config
from amt.sel_runner import Experiment

if __name__ == "__main__":
    for n in [20]:#5,10,15,20
        for mode in ["beta.med"]:# "opt", "beta.med", "ts.5", "ts", "equal", "beta", "means", "mean.slow"
            for m in [1, 0]: # 0, 1, 2, int(n/2)
                for p in [0.5]: #[0.05, 0.2, 0.35, 0.5, 0.6, 0.7, 0.85]:
                    for alpha_c in [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]: # 0.01, 0.05, 0.1    
                        conf = Config(
                            n = n,
                            m = m,
                            sample_size = 2000,
                            initial_size = 10,
                            reps = 2500,
                            common_p = p,
                            p_diff = 0.1,
                            selection_mode = mode,
                            alpha_c = alpha_c
                        )


                        exp = Experiment(conf=conf, save_path = f"./exp_results/coin_res/hyperpar_alpha/alpha{alpha_c}")
                        exp.run_parallel()
                        exp.save()
