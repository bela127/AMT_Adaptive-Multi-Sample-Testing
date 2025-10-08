from os import makedirs

import numpy as np
from matplotlib import pyplot as plt

from amt.configuration import Config

if __name__ == "__main__":
    load_path = "./coin_res"
    save_path = "./coin_plot/coin_selection"
    raw_data = {}
    ref_data = {}
    data = {}
    for n in [20]:#
        for sel_mode in ["ts", "equal", "beta", "mean.slow","beta.med"]:# "ts.5", "ts", "equal", "beta", "means", "mean.slow"
            conf = Config(
                n = n,
                m = 1,
                sample_size = 2000,
                initial_size = 10,
                reps = 10000,
                common_p = 0.5,
                p_diff = 0.05,
                selection_mode = sel_mode,
                coin_weights = "posdif",
            )
            name = conf.get_sel_name()
            contingency = np.load(f"{load_path}/contingency_{name}.npy")

            index = np.arange(contingency.shape[3])
            sizes = np.arange(0,contingency.shape[3])

            coin_sum = np.sum(contingency, axis=1)
            total_sum = np.sum(contingency, axis=(1,2))

            coin_percent = coin_sum / total_sum[:,None,:]


            max_sel = np.max(coin_percent[:,:19], axis = 1)

            max_ratio = max_sel / coin_percent[:,19]
            max_ratio_average = np.mean(max_ratio, axis = 0)

            plt.plot(sizes,max_ratio_average, label=f"{conf.selection_mode}, ratio")

    plt.xlabel("sample size")
    plt.ylabel("selection ratio")
    #plt.ylim(0,0.8)
    plt.legend()
    plt.title(f"Selection ratio for {conf.selection_mode}")
    #plt.savefig(f"./coin_plot/select__{file_name}.png")
    

    makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/coin.selection_{conf.get_sel_name}.png")
    plt.show()