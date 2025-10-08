from os import makedirs


from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

from amt.configuration import Config

load_path = "./stat_res/H0"
conf = Config(
                n = 5,
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

nbins = 50
hist_range = (0,np.max(stat_values))
heatmap = np.empty((nbins, stat_values.shape[1]))
for i in range(stat_values.shape[1]):
    heatmap[:,i] = np.histogram(stat_values[:,i], bins=nbins, range=hist_range, density=True)[0]


print(heatmap.shape)

extent = (sizes[0], sizes[-1], hist_range[0], hist_range[-1])

plt.imshow(heatmap, extent=extent, origin='lower', aspect='auto')
plt.xlabel('Iterations')
plt.ylabel(rf"$\chi^2$ Test Statistic")
plt.colorbar().set_label("pdf")
plt.show()

cum = np.cumsum(heatmap, axis=0)
cum = cum / cum[-1,:] #Needed because of rounding errors
plt.imshow(cum, extent=extent, origin='lower', aspect='auto')
plt.xlabel('Iterations')
plt.ylabel(rf"$\chi^2$ Test Statistic")
plt.colorbar().set_label("cdf")
plt.show()


plt.imshow(heatmap, extent=extent, origin='lower', aspect='auto')
plt.xlabel('Iterations')
plt.ylabel(rf"$\chi^2$ Test Statistic")
plt.colorbar().set_label("pdf")
plt.title(rf"Null Distribution of $\chi^2$ Test Statistic for {conf.n} Coins")
print(cum.shape)

for sig in conf.significance:
    index = (cum >= (1-sig)).argmax(axis=0)
    print(index.shape)
    scaled = index * (hist_range[-1] - hist_range[0]) / nbins  + hist_range[0]
    plt.plot(sizes, scaled, label=rf"empiric critical value at $\alpha = ${sig}")

plt.legend()
plt.savefig(f"{save_path}/null_dist_all_sample_sizes__{name}.svg", format="svg")
plt.show()

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
    plt.plot(x, chi2.pdf(x, df),'r-', lw=2, alpha=0.6, label=f'chi2 fit {float(df):0.2f}')

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
    plt.savefig(f"{save_path}/null_dist_sample_size{s}__{name}.svg", format="svg")
    plt.show()


sizes = np.arange(stat_values.shape[1])
dfs = []
for i in sizes:
    stat = stat_values[:, i]
    params = chi2.fit(stat, floc = 0, fscale=1)
    dfs.append(params[0])

#fit = 9 * initial_size / sizes + 2 * (1-0.25**(sizes-10))
#fit = 9 * initial_size / (initial_size + 0.5 * (sizes - initial_size)) + 2 * (sizes-initial_size) / (sizes-initial_size + 1)
#fit1 = (n-1) * initial_size / (initial_size + 0.25 * (sizes - initial_size) / 2) + 2 * (1 - initial_size / (initial_size + 1.0 * (sizes - initial_size)/ 2))
#fit2 = (n-1) * initial_size / (initial_size + 0.5 * (sizes - initial_size) / 2) + 2 * (1 - initial_size / (initial_size + 0.5 * (sizes - initial_size)/ 2))
#fit3 = (n-1) * initial_size / (initial_size + 0.33 * (sizes - initial_size) / 2) + 2 * (1 - initial_size / (initial_size + 0.5 * (sizes - initial_size)/ 2))


plt.plot(sizes, dfs, label=rf"DOF of Fitted $\chi^2$")
#plt.plot(sizes,fit1, label="function1")
#plt.plot(sizes,fit2, label="function2")
#plt.plot(sizes,fit3, label="function3")

plt.xlabel("Iterations")
plt.ylabel("Degrees of Freedom (DOF)")
plt.legend()
plt.title(rf"DOF of Fitted $\chi^2$ for {conf.n} Coins")
plt.savefig(f"{save_path}/df__{name}.svg", format="svg")
plt.show()