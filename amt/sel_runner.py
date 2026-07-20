from multiprocessing import Pool, cpu_count
from os import makedirs

import numpy as np


from amt.configuration import Config
from amt.selection import Selections
from amt.coin_weights import coin_weights



class Experiment():

    def __init__(self, conf: Config, save_path = "./coin_res") -> None:
        self.save_path = save_path
        self.conf = conf
        

    def get_data(self):
        rng = np.random.default_rng()
        self.ps = coin_weights[self.conf.coin_weights](self.conf)
        samples = rng.binomial(1,self.ps,size=(self.conf.sample_size, self.conf.reps, self.conf.n))
        if self.conf.hyp == 0:
            swapped = np.moveaxis(samples, 1, 0)
            flattened_per_rep = swapped.reshape(self.conf.reps, -1)
            shuffled_flat = rng.permuted(flattened_per_rep, axis=1)
            shuffled_swapped = shuffled_flat.reshape(self.conf.reps, self.conf.sample_size, self.conf.n)
            samples = np.moveaxis(shuffled_swapped, 0, 1)
        print(samples.shape)
        print(self.conf.get_sel_name())
        return samples
    

    def run_parallel(self):
        samples = self.get_data()
        stat_values = [] 
        with Pool(int(cpu_count())) as pool:
            # call the same function with different data in parallel
            r = 0
            for result in pool.imap_unordered(self.perform_experiment, (samples[:,i,:] for i in range(samples.shape[1]))): #imap_unordered imap
                # report the value to show progress
                print(r)
                stat_values.append(result)
                r+=1
        self.stat_values = stat_values
        return stat_values


    def run(self):
        samples = self.get_data()
        stat_values = [] 
        for i in range(samples.shape[1]):
            result = self.perform_experiment(samples[:,i,:])
            # report the value to show progress
            print(i)
            stat_values.append(result)
        self.stat_values = stat_values
        return stat_values

    def perform_experiment(self, samples):
        selection = Selections(conf=self.conf, samples=samples)
        initial_samples = samples[:self.conf.initial_size,...]

        ones = np.sum(initial_samples, axis=0)
        zeros = initial_samples.shape[0] - ones

        contingency = np.concatenate((ones[None,...], zeros[None,...]), axis=0)


        contingencies = np.empty((2, self.conf.n, self.conf.sample_size-self.conf.initial_size+1))
        contingencies[:,:,0] = contingency
        for i in range(self.conf.initial_size, self.conf.sample_size):
            coin1, coin2 = selection.select(contingency)
            ops1 = samples[i,coin1]
            ops2 = samples[i,coin2]

            contingency[0, coin1] += ops1
            contingency[0, coin2] += ops2
            contingency[1, coin1] += 1 - ops1
            contingency[1, coin2] += 1 - ops2

            contingencies[:,:,i-self.conf.initial_size+1] = contingency
        return contingencies

    
    def save(self):
        makedirs(self.save_path, exist_ok=True)
        selected_coin = np.asarray(self.stat_values, dtype=np.int32)
        name = self.conf.get_sel_name()
        np.save(f"{self.save_path}/contingency_{name}.npy", selected_coin)

class ExperimentReal(Experiment):

    def __init__(self, conf: Config, save_path = "./coin_res/real", data_path = "./real_datasets") -> None:
        self.data_path = data_path
        super().__init__(conf=conf, save_path=save_path)
    
    def get_data(self):
        contingency = self.load_data()
        mean = contingency[0] / np.sum(contingency, axis=0)
        self.conf.n = mean.shape[0]
        self.conf.common_p = np.mean(mean)
        if self.conf.m == 1:
            self.conf.m = mean.shape[0]
            self.conf.p_diff = mean.max() - mean.min()
            self.ps = mean
        if self.conf.m == 0:
            self.ps = np.ones_like(mean)*self.conf.common_p
        samples = np.random.binomial(1,self.ps,size=(self.conf.sample_size, self.conf.reps, self.conf.n))
        print(samples.shape)
        print(self.conf.get_sel_name())
        return samples
    
    
    def load_data(self):
        name: str = self.conf.dataset
        contingency = np.load(f"{self.data_path}/{name}.npy")
        return contingency


