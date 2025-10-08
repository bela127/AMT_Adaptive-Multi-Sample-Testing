from os import makedirs

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta


from amt.configuration import Config

if __name__ == "__main__":
    load_path = "./coin_res"
    save_path = "./coin_plot/coin_selection"
    raw_data = {}
    ref_data = {}
    data = {}
    for n in [20]:#
        conf = Config(
            n = n,
            m = 1,
            sample_size = 2000,
            initial_size = 10,
            reps = 10000,
            common_p = 0.5,
            p_diff = 0.05,
            selection_mode = "beta",
            coin_weights = "posdif",
        )
        name = conf.get_sel_name()
        contingency = np.load(f"{load_path}/contingency_{name}.npy")

        sizes = np.arange(0,contingency.shape[3])

        coin_sum = np.sum(contingency, axis=1)
        total_sum = np.sum(contingency, axis=(1,2))

        coin_percent = coin_sum / total_sum[:,None,:]

        
        index = np.argmax(coin_percent[:,:19], axis=1, keepdims=True)

        max_cont = np.take_along_axis(contingency, np.expand_dims(index, axis=1), axis=2).squeeze(axis=2)

        fake_cont = contingency[:,:,19,:]

        comb_cont = np.sum(contingency, axis=2)

        max_cont_av = np.mean(max_cont, axis=0)
        fake_cont_av = np.mean(fake_cont, axis=0)

        null_cont_fake = comb_cont - fake_cont
        null_cont_max = comb_cont - max_cont

        fake_null_cont_av = np.mean(null_cont_fake, axis=0)
        max_null_cont_av = np.mean(null_cont_max, axis=0)


        for i in [500, 1000, 1500, sizes.shape[0]-1]:
            x = np.arange(0, 1, 0.001)
            y1 = beta.pdf(x, *(max_null_cont_av[...,i]+1))
            y2 = beta.pdf(x,*(fake_null_cont_av[...,i]+1))

            y3 = beta.pdf(x, *(max_cont_av[...,i]+1))
            y4 = beta.pdf(x,*(fake_cont_av[...,i]+1))

            plt.plot(x,y4, label="fake")
            plt.plot(x,y2, label="fake null")
            plt.plot(x,y3, label="max")
            plt.plot(x,y1, label="max null")
            
            plt.xlabel("$p_n$")
            plt.ylabel("pdf")
            plt.xlim(0.2,0.8)
            plt.legend()
            plt.title(f"Selection distribution for i={i}")
            
            plt.show()