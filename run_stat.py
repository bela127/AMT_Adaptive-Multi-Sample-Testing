from amt.configuration import Config
from amt.test_runner import StatisticExperiment

if __name__ == "__main__":
    for n in [20]: # [5, 10, 15, 20]
        for sel_mode in ["beta.med"]:# "opt", "ts.5", "ts", "equal", "beta" ,"means", "mean.slow", "beta.med"
            for test_mode in ["eval"]:#"chi2","kw", "beta", "mean"
                for m in [0]: # [0, 1, 2, int(n/2)]
                    for hyp in [0, 1]:
                        conf = Config(
                            n = n,
                            m = m, 
                            sample_size = 2000,
                            initial_size = 10,
                            reps = 2500,
                            common_p = 0.5,
                            p_diff = 0.15,
                            hyp=hyp,
                            selection_mode = sel_mode,
                            test_mode = test_mode,
                            coin_weights = "posdif"
                        )


                    exp = StatisticExperiment(conf=conf, load_path="./coin_res/p15", save_path="./stat_res")
                    exp.run_parallel()
                    exp.save()