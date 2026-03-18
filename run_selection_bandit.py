import numpy as np

from amt.configuration import Config
from amt.sel_runner import ExperimentBandit

if __name__ == "__main__":
    for n in [20]:#5,10,15,20
            for m in [1]: # 0, 1, 2, int(n/2)
                conf = Config(
                    n = n,
                    m = m,
                    sample_size = 4000,
                    initial_size = 10,
                    reps = 1000,
                    common_p = 0.5,
                    p_diff = 0.15,
                    selection_mode = "bandit",
                    test_mode="bandit"
                )


                exp = ExperimentBandit(conf=conf)
                exp.run_parallel()
                exp.save()
