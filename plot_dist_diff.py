from os import makedirs

import numpy as np
import matplotlib.pyplot as plt

from amt.configuration import Config
from amt.utils import translate_names

load_path = "./stat_res"
save_path = f"./coin_plot/diff"
conf = Config(
                n = 10,
                m = 1,
                sample_size = 2000,
                initial_size = 10,
                reps = 10000,
                common_p = 0.5,
                p_diff = 0.05,
                selection_mode = "beta.med",
                test_mode="beta",
                coin_weights = "posdif",
            )
stat_name = conf.get_test_name()
stat_values = np.load(f"{load_path}/H1/teststat_{stat_name}.npy")

null_conf = conf.clone()
null_conf.m = 0
null_name = null_conf.get_test_name()
null_values = np.load(f"{load_path}/H0/teststat_{null_name}.npy")

makedirs(save_path, exist_ok=True)

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

plt.figure(figsize=(6, 3))
plt.imshow(diff_map, extent=extent, origin='lower', aspect='auto', cmap='bwr')

test_name, sel_name = translate_names(conf.test_mode, conf.selection_mode)
plt.xlabel('Iterations')
plt.ylabel(f"{test_name} Test Statistic")
plt.title(rf"Difference Between $H_0$ and $H_1$ Distribution{'\n'}of {test_name} Test Statistic under {sel_name} Sampling for {conf.n} Coins")
plt.colorbar().set_label("pdf difference")
plt.savefig(f"{save_path}/diff_dist__{stat_name}.svg", format="svg", bbox_inches='tight')
plt.show()

abs_map = np.abs(diff_map)
diff_count = np.sum(abs_map, axis=0)

sum_count = np.sum(heatmap + heatmap_null, axis=0)
diff_prop=diff_count/sum_count

plt.figure(figsize=(6, 3))
plt.plot(sizes, diff_prop)
plt.title(rf"TVD Between $H_0$ and $H_1${'\n'}of {test_name} Test Statistic under {sel_name} Sampling for {conf.n} Coins")
plt.xlabel('Iterations')
plt.ylabel(rf"TVD between $H_0$ and $H_1$")
plt.savefig(f"{save_path}/diff_dist_dist__{stat_name}.png")
plt.show()


sizes = [0, 25, 50, 100, 250, 500, 1000, 1500, 1990]
for i in sizes:
    plt.figure(figsize=(6, 3))
    stat = stat_values[:, i]
    dens, pos, _ = plt.hist(stat, density=True, bins=nbins)

    stat = null_values[:, i]
    dens, pos, _ = plt.hist(stat, density=True, bins=nbins)
    for sig, c in zip(conf.significance, ["red", "orange", "yellow"]):
        index = int((1-sig)*stat.shape[0])
        crit_val = np.partition(stat,index)[index]
        plt.vlines(crit_val, [0], [np.max(dens)], colors=c, label=rf"empiric critical value at $\alpha = ${sig}")

    plt.xlabel(f"{test_name} Test Statistic")
    plt.ylabel("pdf")

    plt.title(rf"Distributions of {test_name} Test Statistic{'\n'}for {conf.n} Coins at Iterations = {i}")
    plt.legend()
    plt.savefig(f"{save_path}/dist_diff_sample_size{i}__{stat_name}.svg", format="svg", bbox_inches='tight')
    plt.show()