from amt.configuration import Config
from amt.test_runner import TestExperiment
from amt.tests import Test

test = Test()

if __name__ == "__main__":
    for n in [20]:
        for sel_mode in ["beta.med"]:
            for test_mode in ["betabinom.pmf"]:
                for m in [1, 2, 4, 6, 8, 10, 15]:
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
                    )


                    exp = TestExperiment(conf=conf, load_path="./exp_results/coin_res/vari_m", save_path="./exp_results/test_res/vari_m")
                    exp.run_parallel()
                    exp.save()