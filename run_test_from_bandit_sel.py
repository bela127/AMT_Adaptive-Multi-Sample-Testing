from amt.configuration import Config
from amt.test_runner import TestExperiment
from amt.tests import Test
from amt.bandit_bounds import bandit_bounds


test = Test()

if __name__ == "__main__":
    for n in [20]: # [5, 10, 15, 20]
        for mode in ["bandit.max", "bandit.ratio"]:# "opt", "beta.med", "ts.5", "ts", "equal", "beta", "means", "mean.slow"
            for bandit in bandit_bounds.keys():
                for test_mode in ["hoeffding.variance.infinite", "lil.variance", "glrt.horizon", "one.vs.rest.beta.mixture", "beta.mixture.infinite", "betting.e.variable", "bayesian.beta", "betabinom.pmf"]:
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
                                bandit_kind = bandit,
                                test_mode = test_mode,
                                coin_weights = "posdif"
                            )


                            exp = TestExperiment(conf=conf, load_path="./exp_results/coin_res/bandit_sel", save_path="./exp_results/test_res/bandit_sel")
                            exp.run_parallel()
                            exp.save()