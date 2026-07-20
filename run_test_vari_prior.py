from amt.configuration import Config
from amt.test_runner import TestExperiment
from amt.tests import Test

test = Test()

common_p = 0.5
p_diff = 0.075

if __name__ == "__main__":
    for n in [20]:
        for sel_mode in ["beta.med"]:
            for test_mode in ["betabinom.pmf"]:
                for m in [1,0]:
                    for prior in [(4,2),((common_p*n+p_diff)/n*3,(common_p*n-p_diff)/n*3),(2,2), (1,1), (0.5,0.5), (0.25,0.25), (0,0)]:
                        conf = Config(
                            n = n,
                            m = m,
                            hyp=m,
                            sample_size = 2000,
                            initial_size = 10,
                            reps = 2500,
                            common_p = common_p,
                            p_diff = p_diff,
                            selection_mode = sel_mode,
                            test_mode = test_mode,
                            prior=prior
                        )


                        exp = TestExperiment(conf=conf, load_path=f"./exp_results/coin_res/vari_prior/prior_{prior[0]:.2f}_{prior[1]:.2f}", save_path=f"./exp_results/test_res/vari_prior/prior_{prior[0]:.2f}_{prior[1]:.2f}")
                        exp.run_parallel()
                        exp.save()