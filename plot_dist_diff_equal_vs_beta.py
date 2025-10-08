from os import makedirs

import numpy as np
import matplotlib.pyplot as plt

from amt.configuration import Config
from amt.utils import translate_names, translate_test_colors

load_path = "./stat_res"
save_path = f"./coin_plot/diff_equal_vs_beta"

if __name__ == "__main__":
    makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.rcParams["lines.dashed_pattern"] = (5,10)
    for sel_mode in ["equal", "beta.med"]:
        for test_mode in ["chi2", "beta"]:#"mean", "chi2","kw", "beta", "betabinom.comb"
            conf = Config(
                            n = 10,
                            m = 1,
                            sample_size = 2000,
                            initial_size = 10,
                            reps = 10000,
                            common_p = 0.5,
                            p_diff = 0.05,
                            selection_mode = sel_mode,
                            test_mode=test_mode,
                            coin_weights = "posdif",
                        )
            stat_name = conf.get_test_name()
            stat_values = np.load(f"{load_path}/H1/teststat_{stat_name}.npy")

            null_conf = conf.clone()
            null_conf.m = 0
            null_name = null_conf.get_test_name()
            null_values = np.load(f"{load_path}/H0/teststat_{null_name}.npy")

            print(stat_values.shape)

            sizes = np.arange(stat_values.shape[1])

            nbins = 50
            hist_range = (min(np.min(stat_values), np.min(null_values)),max(np.max(stat_values), np.max(null_values)))

            heatmap = np.empty((nbins, stat_values.shape[1]))
            for i in range(stat_values.shape[1]):
                heatmap[:,i] = np.histogram(stat_values[:,i], bins=nbins, range=hist_range, density=True)[0]

            extent = (sizes[0], sizes[-1], hist_range[0], hist_range[-1])

            heatmap_null = np.empty((nbins, null_values.shape[1]))
            for i in range(null_values.shape[1]):
                heatmap_null[:,i] = np.histogram(null_values[:,i], bins=nbins, range=hist_range, density=True)[0]

            diff_map = heatmap - heatmap_null

            abs_map = np.abs(diff_map)
            diff_count = np.sum(abs_map, axis=0)

            sum_count = np.sum(heatmap + heatmap_null, axis=0)
            diff_prop=diff_count/sum_count

            test_name, sel_name = translate_names(conf.test_mode, conf.selection_mode)

            line_color = translate_test_colors(test_mode)
            if sel_mode == "equal":
                line_style = 'dashed'
                plt.plot(sizes, diff_prop, label=f"{sel_name} vs", color=line_color, linestyle=line_style)
            else:
                line_style = 'solid'
                plt.plot(sizes, diff_prop, label=f"{sel_name} selection for {test_name} statistic", color=line_color, linestyle=line_style)
            
            plt.title(rf"TVD Between $H_0$ and $H_1${'\n'} Equal vs Beta Sampling for {conf.n} Coins")
            plt.xlabel('Iterations')
            plt.ylabel(rf"TVD between $H_0$ and $H_1$")
            plt.legend(ncol=2, columnspacing=0.5)
    plt.savefig(f"{save_path}/diff_dist_dist__{stat_name}.svg", format="svg", bbox_inches='tight')
    plt.show()