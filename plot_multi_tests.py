import numpy as np
import matplotlib.pyplot as plt
from os import makedirs

from amt.configuration import Config


if __name__ == "__main__":
    for n in [5, 10, 15, 20]:#5, 10, 15, 20
        for sel_mode in ["beta.med"]:# "opt", "beta.med", "ts.5", "ts", "equal", "beta", "means", "mean.slow"
            conf = Config(
                n = n,
                m = 0,
                sample_size = 2000,
                initial_size = 10,
                reps = 10000,
                common_p = 0.5,
                p_diff = 0.05,
                selection_mode = sel_mode,
                test_mode = "betabinom.comb"
            )

            tests = np.load(f"./multi_stat_res/H0/multi.teststat_{conf.get_test_name()}.npy")

            for sig_index, alpha in enumerate(conf.significance):


                index = np.arange(tests.shape[2])

                reject_count = np.count_nonzero(tests, axis=3)
                power = reject_count/tests.shape[3]


                fig, ax = plt.subplots()

                ax.plot(index[None,...].T, power[sig_index].T, linestyle = "dotted", label=[f"coin {k}" for k in range(1,n+1)])

                ax.plot([0,tests.shape[2]], [alpha/n,alpha/n], label=f"emp sig={alpha}", color='black')

                plt.xlabel("Iterations")
                plt.ylabel("Type I Error")
                plt.ylim(0, alpha/n + 0.4*alpha/n)
                plt.legend(ncol=4)
                plt.title(rf'''Single test type I error
                at corrected significance level $\alpha_n={alpha/n:0.4f}$ for {n} coins''')
                makedirs("./coin_plot/alpha", exist_ok=True)
                plt.savefig(f"./coin_plot/alpha/power_single__teststat_test.mode-{conf.test_mode}_sel.mode-{conf.selection_mode}_coins-{n}_chance-{conf.common_p*100:07.4f}_samplemax-{conf.sample_size}_initialsize-{conf.initial_size}_reps-{conf.reps}_sig-{alpha}.svg", format="svg", bbox_inches='tight', transparent="True", pad_inches=0)
                #plt.show()
                plt.close()