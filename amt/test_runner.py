from multiprocessing import Pool, cpu_count
from os import makedirs

import numpy as np
from numpy._typing._array_like import NDArray

from amt.configuration import Config
from amt.test import Tests

class Experiment():
    def __init__(self, conf: Config, save_path = "./stat_res", load_path = "./coin_res") -> None:
        self.conf = conf
        self.save_path = save_path
        self.load_path = load_path
        self.test = Tests(conf=self.conf)
        print(self.conf.get_test_name())

    def load_data(self):
        contingency = np.load(f"{self.load_path}/contingency_{self.conf.get_sel_name()}.npy")
        contingency = np.transpose(contingency,axes=(1,2,3,0))
        print(contingency.shape)
        return contingency
    
    def calc_splits(self, contingency):
        part = 1
        if contingency.shape[3] // 1000 >10:
            part = 1000
        elif contingency.shape[3] // 500 >10:
            part = 500
        elif contingency.shape[3] // 250 >5:
            part = 250
        elif contingency.shape[3] // 100 >5:
            part = 100
        elif contingency.shape[3] // 50 > 5:
            part = 50
        elif contingency.shape[3] // 25 > 5:
            part = 25
        elif contingency.shape[3] // 10 > 5:
            part = 10
        elif contingency.shape[3] // 5 > 5:
            part = 5
        elif contingency.shape[3] // 3 > 5:
            part = 3

        return np.array_split(contingency, part, axis=3), part

    def run_parallel(self):
        contingency = self.load_data()

        stat_values = [] 
        pool_size = int((cpu_count()+0.5)/1.1)
        with Pool(pool_size) as pool:
            splits, part = self.calc_splits(contingency)
            
            b = 0
            print(f"start b={b}, splits={part}, size={contingency.shape[3]//part} and {contingency.shape[3]//part + 1}, on {pool_size} tasks")

            # call the same function with different data in parallel
            for result in pool.imap_unordered(self.run_test, splits): #imap_unordered imap
                # report the value to show progress
                b+=1
                print(b)
                stat_values.append(result)
        stat_values = np.concatenate(stat_values, axis=0)
        self.stat_values = stat_values
        return stat_values

    def run(self):
        contingency = self.load_data()
        stat_values = self.run_test(contingency)
        self.stat_values = stat_values
        return stat_values
    
    def run_test(self, contingency):
        i, r = contingency.shape[2], contingency.shape[3]
        test_results = np.empty((r, i))
        for idx_i in range(i):
            for idx_r in range(r):
                test_results[idx_r, idx_i] = self.test.test(contingency[:, :, idx_i, idx_r])
        return test_results
    
    def save(self):
        makedirs(self.save_path, exist_ok=True)
        stat_values = np.asarray(self.stat_values)
        name = self.conf.get_test_name()
        np.save(f"{self.save_path}/teststat_{name}.npy", stat_values)