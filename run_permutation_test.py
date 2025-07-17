from atm.configuration import Config
from atm.permutation_test_runner import Experiment

if __name__ == "__main__":
    for n in [5, 10, 15, 20]:
        for sel_mode in ["ts.5", "ts", "equal", "beta", "means", "mean.slow"]:# 
            for test_mode in ["betabinom.pmf"]:
                for m in [0]:#[0,1]:

                    conf = Config(
                        n = n,
                        m = m,
                        sample_size = 2000,
                        initial_size = 10,
                        reps = 5000,
                        common_p = 0.5,
                        p_diff = 0.05,
                        selection_mode = sel_mode,
                        test_mode = test_mode,
                        coin_weights = "posdif",
                    )

                    null_conf = conf.clone()
                    null_conf.m = 0


                    exp = Experiment(conf=conf, null_conf=null_conf, save_path = "./test_res/H0_test", load_path = "./stat_res/H0_test", null_path = "./stat_res/H0")
                    exp.run_parallel()
                    exp.save()