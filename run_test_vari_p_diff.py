from amt.configuration import Config
from amt.test_runner import TestExperiment
from amt.tests import Test

test = Test()

if __name__ == "__main__":
    for n in [20]: # [5, 10, 15, 20]
        for sel_mode in ["beta.med"]:# "equal", "beta.med"
            for test_mode in ["betabinom.pmf"]: # "betabinom.pmf", "chi2", "kw"
                for m in [1]: # 0, 1, 2, int(n/2)
                    for hyp in [1]:
                        for p_diff in [0.2]:
                            conf = Config(
                                n = n,
                                m = m,
                                hyp=hyp,
                                sample_size = 2000,
                                initial_size = 10,
                                reps = 2500,
                                common_p = 0.5,
                                p_diff = p_diff,
                                selection_mode = sel_mode,
                                test_mode = test_mode,
                                coin_weights = "posdif",
                            )


                            exp = TestExperiment(conf=conf, load_path="./exp_results/coin_res", save_path="./exp_results/test_res")
                            exp.run_parallel()
                            exp.save()