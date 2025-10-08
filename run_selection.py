import numpy as np

from amt.configuration import Config
from amt.sel_runner import Experiment

if __name__ == "__main__":
    for n in [5,10,15,20]:#5,10,15,20
        for mode in ["opt", "beta.med", "ts.5", "ts", "equal", "beta", "means", "mean.slow"]:# "opt", "beta.med", "ts.5", "ts", "equal", "beta", "means", "mean.slow"
            for m in [0, 1]: # 0, 1, 2, int(n/2)
                conf = Config(
                    n = n,
                    m = m,
                    sample_size = 2000,
                    initial_size = 10,
                    reps = 10000,
                    common_p = 0.5,
                    p_diff = 0.05,
                    selection_mode = mode,
                )


                exp = Experiment(conf=conf)
                exp.run_parallel()
                exp.save()
