import numpy as np

from atm.configuration import Config
from atm.sel_runner import Experiment

if __name__ == "__main__":
    for n in [5,10,15,20]:
        for mode in ["ts.5", "ts", "equal", "beta","means", "mean.slow"]:#  

            conf = Config(
                n = n,
                m = 0,
                sample_size = 2000,
                initial_size = 10,
                reps = 2500,
                common_p = 0.5,
                p_diff = 0.05,
                selection_mode = mode,
            )


            exp = Experiment(conf=conf)
            exp.run_parallel()
            exp.save()
