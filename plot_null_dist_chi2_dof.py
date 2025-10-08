from os import makedirs


from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

from amt.configuration import Config

load_path = "./stat_res/H0"

for n in [5,10,15,20]:
    conf = Config(
                n = n,
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

    save_path = f"./coin_plot/chi2/dof"
    makedirs(save_path, exist_ok=True)

    print(stat_values.shape)


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


    plt.plot(sizes, dfs, label=rf"DOF of Fitted $\chi^2$ for {conf.n} Coins")
    #plt.plot(sizes,fit1, label="function1")
    #plt.plot(sizes,fit2, label="function2")
    #plt.plot(sizes,fit3, label="function3")

plt.xlabel("Iterations")
plt.ylabel("Degrees of Freedom (DOF)")
plt.legend()
plt.title(rf"DOF of Fitted $\chi^2$")
plt.savefig(f"{save_path}/df.svg", format="svg")
plt.show()