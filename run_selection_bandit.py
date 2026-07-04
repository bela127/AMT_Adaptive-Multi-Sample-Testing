import numpy as np

from amt.configuration import Config
from amt.sel_runner import Experiment
from amt.bandit_bounds import bandit_bounds

if __name__ == "__main__":
    for n in [20]:#5,10,15,20
        for mode in ["bandit.max", "bandit.ratio"]:# "opt", "beta.med", "ts.5", "ts", "equal", "beta", "means", "mean.slow"
            for bandit in bandit_bounds.keys():
                for m in [1]: # 0, 1, 2, int(n/2)
                    for p in [0.5]:
                        for p_diff in [0.1]:
                            conf = Config(
                                n = n,
                                m = m,
                                common_p = p,
                                p_diff = p_diff,
                                selection_mode = mode,
                                bandit_kind = bandit
                            )


                            exp = Experiment(conf=conf)
                            exp.run_parallel()
                            exp.save()
