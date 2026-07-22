from amt.configuration import Config
from amt.test_runner import TestExperiment
from amt.tests import Test


test = Test()

if __name__ == "__main__":
    for n in [20]: # [5, 10, 15, 20]
        for mode in ["equal", "ts", "ts.5", "means", "mean.slow", "beta.med", "beta.max", "evar"]:# "beta.med", "beta.max", "evar"
            for test_mode in ["kl.horizon", "bayesian.e.variable","hoeffding.variance.infinite", "lil.variance", "glrt.horizon", "one.vs.rest.beta.mixture", "beta.mixture.infinite", "betting.e.variable", "betabinom.pmf"]:
                for m in [1]: # 0, 1, 2, int(n/2)
                    for p in [0.5]:
                        conf = Config(
                            n = n,
                            m = m,
                            hyp=m,
                            sample_size = 2000,
                            initial_size = 10,
                            reps = 2500,
                            common_p = p,
                            p_diff = 0.1,
                            selection_mode = mode,
                            test_mode = test_mode,
                            coin_weights = "posdif"
                        )


                        exp = TestExperiment(conf=conf, load_path="./exp_results/coin_res/non_bandit_selection", save_path="./exp_results/test_res/non_bandit_selection")
                        exp.run_parallel()
                        exp.save()