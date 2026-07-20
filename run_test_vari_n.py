from amt.configuration import Config
from amt.test_runner import TestExperiment
from amt.tests import Test

test = Test()

if __name__ == "__main__":
    for n in [5, 10, 15, 20, 30, 40]:
        for sel_mode in ["beta.med"]:
            for test_mode in ["betabinom.pmf"]:
                for m in [1]:
                    conf = Config(
                        n = n,
                        m = m,
                        hyp=m,
                        sample_size = 2000,
                        initial_size = 10,
                        reps = 2500,
                        common_p = 0.5,
                        p_diff = 0.05,
                        selection_mode = sel_mode,
                        test_mode = test_mode,
                    )


                    exp = TestExperiment(conf=conf, load_path=f"./exp_results/coin_res/vari_n", save_path=f"./exp_results/test_res/vari_n")
                    exp.run_parallel()
                    exp.save()