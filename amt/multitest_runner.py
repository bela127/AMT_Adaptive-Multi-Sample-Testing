from multiprocessing import Pool, cpu_count
from os import makedirs

import numpy as np

from amt.configuration import Config
from amt.multitest import MultiTests

class Experiment():
    def __init__(self, conf: Config, save_path = "./multi_stat_res", load_path = "./coin_res", batch_size=1) -> None:
        self.conf = conf
        self.save_path = save_path
        self.load_path = load_path
        self.test = MultiTests(conf=self.conf)
        self.batch_size = batch_size
        print(self.conf.get_test_name())

    def load_data(self):
        contingency = np.load(f"{self.load_path}/contingency_{self.conf.get_sel_name()}.npy")
        contingency = np.transpose(contingency,axes=(1,2,3,0))
        #contingency=contingency[:,:,:10,:10]
        print(contingency.shape)
        return contingency
    
    def calc_splits(self, contingency):
        part = 4
        return np.array_split(contingency, part, axis=2), part

    def run_parallel(self):
        contingency = self.load_data()
        part = contingency.shape[2] // self.batch_size
        splits = np.array_split(contingency, part, axis=2)

        stat_values = [] 
        pool_size = int((cpu_count()+0.5)/1.5)
        with Pool(pool_size) as pool:
            
            b = 0
            print(f"start b={b}, splits={part}, size={contingency.shape[2]//part} and {contingency.shape[2]//part + 1}, on {pool_size} tasks")

            # call the same function with different data in parallel
            for result in pool.imap_unordered(self.test.test, splits): #imap_unordered imap
                # report the value to show progress
                b+=1
                print(b)
                stat_values.append(result)
        stat_values = np.concatenate(stat_values, axis=2)
        self.stat_values = stat_values
        return stat_values

    def run(self):
        contingency = self.load_data()
        stat_values = self.test.test(contingency)
        self.stat_values = stat_values
        return stat_values
    
    def save(self):
        makedirs(self.save_path, exist_ok=True)
        stat_values = np.asarray(self.stat_values).astype(bool)
        name = self.conf.get_test_name()
        np.save(f"{self.save_path}/multi.teststat_{name}.npy", stat_values)