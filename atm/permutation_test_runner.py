from multiprocessing import Pool, cpu_count
from os import makedirs

import numpy as np
from numpy._typing._array_like import NDArray

from atm.configuration import Config

class Experiment():
    def __init__(self, conf: Config, null_conf: Config, save_path = "./test_res", load_path = "./stat_res", null_path = "./stat_res") -> None:
        self.conf = conf
        self.null_conf = null_conf
        self.save_path = save_path
        self.load_path = load_path
        self.null_path = null_path
        print(self.conf.get_test_name())

    def load_data(self):
        teststat = np.load(f"{self.load_path}/teststat_{self.conf.get_test_name()}.npy")
        print(teststat.shape)
        nullstat = np.load(f"{self.null_path}/teststat_{self.null_conf.get_test_name()}.npy")
        print(nullstat.shape)
        return teststat, nullstat

    def run_parallel(self):
        print("Not required, using run() instead.")
        return self.run()

    def run(self):
        teststat, nullstat = self.load_data()

        indexes = np.ceil((1-np.asarray(self.conf.significance))*nullstat.shape[0]).astype(int)
        crit_vals = np.partition(nullstat,indexes, axis=0)[indexes,:]

        rejected = teststat[None,...]>crit_vals[:,None,:]
        reject_count = np.count_nonzero(rejected, axis=1)
        power = reject_count/teststat.shape[0]

        self.rejected = rejected
        self.power = power
        return rejected, power
    
    def save(self):
        makedirs(self.save_path, exist_ok=True)
        rejected = np.asarray(self.rejected)
        name = self.conf.get_test_name()
        np.save(f"{self.save_path}/reject_{name}.npy", rejected)

        power = np.asarray(self.power)
        name = self.conf.get_test_name()
        np.save(f"{self.save_path}/power_{name}.npy", power)