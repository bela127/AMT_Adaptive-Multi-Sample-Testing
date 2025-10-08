from os import makedirs

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom, betabinom

from amt.configuration import Config
load_path = "./coin_res/H0"
conf = Config(
                n = 10,
                m = 0,
                sample_size = 2000,
                initial_size = 10,
                reps = 10000,
                common_p = 0.5,
                p_diff = 0.05,
                selection_mode = "beta.med",
                coin_weights = "posdif",
            )
name = conf.get_sel_name()
contingency = np.load(f"{load_path}/contingency_{name}.npy")

contingency_t = np.transpose(contingency, (0, 3, 2, 1))

contingency_t = contingency_t[:,:,:,:]

merged_contingency = np.reshape(contingency_t, shape=(-1, contingency_t.shape[2],contingency_t.shape[3]))

sample_size = np.sum(merged_contingency, axis=-1)

sort_index = np.argsort(sample_size[...,0], axis=0)
sorted_sample_size = np.take_along_axis(sample_size, sort_index[...,None], axis=0)
sorted_contingency = np.take_along_axis(merged_contingency, sort_index[...,None,None], axis=0)

#for i in range(sorted_sample_size.shape[1]):
unique_sample_size, unique_index = np.unique(sorted_sample_size[...,0], return_index=True)
splits = np.split(sorted_contingency, unique_index[1:], axis=0)
for j, split in zip(unique_index, splits):
    throws = sorted_sample_size[j]
    #print(i, throws)
    #for iteration in [10, 50, 200, 500, 1000, 1500]:
        #if i == iteration:
    if throws[0]  in [10, 50, 200, 500, 1000, 1500]:
        fr = binom(throws[0], conf.common_p)
        x = np.arange(0, throws[0] + 1)
        y = fr.pmf(x)

        not_coin = split[:,1:]
        heads_not = np.sum(not_coin[...,0], axis=1)
        tails_not = np.sum(not_coin[...,1], axis=1)

        head_sort_index = np.argsort(heads_not, axis=0)
        sorted_heads = np.take_along_axis(heads_not, head_sort_index, axis=0)
        sorted_tails = np.take_along_axis(tails_not, head_sort_index, axis=0)

        mid = int(sorted_heads.shape[0]/2)
        low = int(sorted_heads.shape[0]/4)

        head_low = sorted_heads[low]
        tail_low = sorted_tails[low]
        head_med = sorted_heads[mid]
        tail_med = sorted_tails[mid]

        betabinomfr_low = betabinom(throws[0], 1+head_low, 1+tail_low)
        betabinomfr_med = betabinom(throws[0], 1+head_med, 1+tail_med)
        betabinomfr_up = betabinom(throws[0], 1+tail_low, 1+head_low)


        y_low = betabinomfr_low.pmf(x)
        y_med = betabinomfr_med.pmf(x)
        y_up = betabinomfr_up.pmf(x)


        plt.title(f"Binomial PMF for {throws[0]} Throws")
        plt.bar(x, y_low, label=f"BetaBin low PMF n={throws[0]}")
        plt.bar(x, y_med, label=f"BetaBin med PMF n={throws[0]}")
        plt.bar(x, y_up, label=f"BetaBin up PMF n={throws[0]}")
        plt.bar(x, y, label=f"Binomial PMF n={throws[0]}")
        plt.hist(split[:,0,0], bins=np.arange(0, throws[0] + 2) - 0.5, density=True, alpha=0.5, label=f"Observed Frequencies n={throws[0]}")
        plt.xlabel("Number of Heads")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()
        plt.clf()
        plt.close()
