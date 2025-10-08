from os import makedirs


from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

from amt.configuration import Config

load_path = "./stat_res/H0"
conf = Config(
                n = 20,
                m = 0,
                sample_size = 2000,
                initial_size = 10,
                reps = 10000,
                common_p = 0.5,
                p_diff = 0.05,
                selection_mode = "beta.med",
                test_mode="chi2",
                coin_weights = "posdif",
            )
name = conf.get_test_name()
stat_values = np.load(f"{load_path}/teststat_{name}.npy")

save_path = f"./coin_plot/chi2/{conf.n}coins"
makedirs(save_path, exist_ok=True)

print(stat_values.shape)

sizes = np.arange(stat_values.shape[1])

plt.figure(figsize=(5, 2.5))
nbins = 50
sizes = [0, 25, 50, 100, 250, 500, 1000, 1500, 1990]
for s in sizes:
    stat = stat_values[:, s]
    dens, pos, _ = plt.hist(stat, density=True, bins=nbins)
    #for sig, c in zip(conf.significance, ["red", "orange", "yellow"]):
    #    index = int((1-sig)*stat.shape[0])
    #    crit_val = np.partition(stat,index)[index]
    #    plt.vlines(crit_val, [0], [np.max(dens)], colors=c, label=f"emp. critical value {sig=}")

    params = chi2.fit(stat, floc = 0, fscale=1)
    df = params[0]
    x = np.linspace(chi2.ppf(0.001, df),chi2.ppf(0.999, df), 100)
    plt.plot(x, chi2.pdf(x, df),'r-', lw=2, alpha=0.6, label=f'chi2 fit DOF={float(df):0.2f}')

    #for sig, c in zip(conf.significance, ["red", "orange", "yellow"]):
    #    plt.vlines([chi2.ppf(1-sig, df)], [0], [np.max(dens)], colors=c, linestyles = "dashed", label=f"fit critical value {sig=}")


    cdf = (conf.n-1)
    x = np.linspace(chi2.ppf(0.001, cdf),chi2.ppf(0.999, cdf), 100)
    plt.plot(x, chi2.pdf(x, cdf),'g-', lw=2, alpha=0.6, label=f'chi2 classic DOF={float(cdf):0.2f}')

    #cdf = 2
    #x = np.linspace(chi2.ppf(0.001, cdf),chi2.ppf(0.999, cdf), 100)
    #plt.plot(x, chi2.pdf(x, cdf),'g-', lw=2, alpha=0.6, label=f'chi2 end df={cdf}')

    plt.xlabel(rf"$\chi^2$ Test Statistic")
    plt.ylabel("pdf")

    plt.title(rf"Null Distribution of $\chi^2$ Test Statistic{'\n'}for {conf.n} Coins at Iterations = {s}")
    plt.legend()
    plt.savefig(f"{save_path}/null_dist_sample_size{s}__{name}.svg", format="svg", bbox_inches='tight')
    plt.clf()
    #plt.show()