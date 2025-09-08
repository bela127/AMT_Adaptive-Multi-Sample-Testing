from os import makedirs

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom

from atm.configuration import Config
load_path = "./coin_res/H0"
conf = Config(
                n = 10,
                m = 0,
                sample_size = 2000,
                initial_size = 10,
                reps = 5000,
                common_p = 0.5,
                p_diff = 0.05,
                selection_mode = "beta.med",
                coin_weights = "posdif",
            )
name = conf.get_sel_name()
contingency = np.load(f"{load_path}/contingency_{name}.npy")

contingency_t = np.transpose(contingency, (0, 2, 3, 1))

contingency_t = contingency_t[:,:,:,:]

merged_contingency = np.reshape(contingency_t, shape=(-1, contingency_t.shape[3]))

sample_size = np.sum(merged_contingency, axis=1)

sort_index = np.argsort(sample_size, axis=0)
sorted_sample_size = np.take_along_axis(sample_size, sort_index, axis=0)
sorted_contingency = np.take_along_axis(merged_contingency, sort_index[:,None], axis=0)

#for i in range(sorted_sample_size.shape[1]):
unique_sample_size, unique_index = np.unique(sorted_sample_size, return_index=True)
splits = np.split(sorted_contingency, unique_index[1:], axis=0)
for j, split in zip(unique_index, splits):
    throws = sorted_sample_size[j]
    #print(i, throws)
    #for iteration in [10, 50, 200, 500, 1000, 1500]:
        #if i == iteration:
    if throws  in [10, 50, 200, 500, 1000, 1500]:
        fr = binom(throws, conf.common_p)
        x = np.arange(0, throws + 1)
        y = fr.pmf(x)
        plt.title(f"Binomial PMF for {throws} Throws")
        plt.bar(x, y, label=f"Binomial PMF n={throws}")
        plt.hist(split[:,0], bins=np.arange(0, throws + 2) - 0.5, density=True, alpha=0.5, label=f"Observed Frequencies n={throws}")
        plt.xlabel("Number of Heads")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()
