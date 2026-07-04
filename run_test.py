from amt.configuration import Config
from amt.test_runner import TestExperiment
from amt.tests import Test

test = Test()

if __name__ == "__main__":
    for n in [20]: # [5, 10, 15, 20]
        for sel_mode in ["beta.med"]:# "opt", "ts.5", "ts", "equal", "beta" ,"means", "mean.slow", "beta.med"
            for test_mode in test.test_modes.keys(): #["bayesian.beta.loo"]:#"betting.e.variable"]: # 
                for m in [0, 1]: # 0, 1, 2, int(n/2)
                    for p in [0.05, 0.2, 0.35, 0.5, 0.6, 0.7, 0.85]:
                        conf = Config(
                            n = n,
                            m = m,
                            hyp=m,
                            sample_size = 2000,
                            initial_size = 10,
                            reps = 2500,
                            common_p = p,
                            p_diff = 0.1,
                            selection_mode = sel_mode,
                            test_mode = test_mode,
                            coin_weights = "posdif"
                        )


                        exp = TestExperiment(conf=conf, load_path="./coin_res", save_path="./test_res")
                        exp.run_parallel()
                        exp.save()