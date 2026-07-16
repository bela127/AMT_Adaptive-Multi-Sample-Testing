from amt.configuration import Config
from amt.test_runner import TestExperiment
from amt.tests import Test

test = Test()

if __name__ == "__main__":
    for n in [20]: # [5, 10, 15, 20]
        for sel_mode in ["beta.med"]:# "opt", "ts.5", "ts", "equal", "beta" ,"means", "mean.slow", "beta.med"
            for test_mode in ["betabinom.pmf"]: #["bayesian.beta.loo"]:#"betting.e.variable"]: # 
                for m in [0, 1]: # 0, 1, 2, int(n/2)
                    for p in [0.5]:
                        for alpha_c in [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]:
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
                                coin_weights = "posdif",
                                alpha_c = alpha_c
                            )


                            exp = TestExperiment(conf=conf, load_path=f"./exp_results/coin_res/hyperpar_alpha/alpha{alpha_c}", save_path=f"./exp_results/test_res/hyperpar_alpha/alpha{alpha_c}")
                            exp.run_parallel()
                            exp.save()