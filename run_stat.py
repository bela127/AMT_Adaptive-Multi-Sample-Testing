from atm.configuration import Config
from atm.test_runner import Experiment

if __name__ == "__main__":
    for n in [5, 10, 15, 20]:
        for sel_mode in ["ts.5", "ts", "equal", "beta", "means", "mean.slow"]:# 
            for test_mode in ["chi2","kw", "beta", "mean"]:
                for m in [0]:#[0,1]:

                    conf = Config(
                        n = n,
                        m = m,
                        sample_size = 2000,
                        initial_size = 10,
                        reps = 2500,
                        common_p = 0.5,
                        p_diff = 0.05,
                        selection_mode = sel_mode,
                        test_mode = test_mode,
                        coin_weights = "posdif"
                    )


                    exp = Experiment(conf=conf, load_path="./coin_res/H0_test", save_path="./stat_res/H0_test")
                    exp.run_parallel()
                    exp.save()