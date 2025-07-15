from multiprocessing import Pool, cpu_count
from os import makedirs

import numpy as np


from atm.configuration import Config, coin_weights
from atm.selection import Selections



class Experiment():

    def __init__(self, conf: Config, save_path = "./coin_res") -> None:
        self.save_path = save_path
        self.conf = conf
        print(self.conf.get_sel_name())
        

    def get_data(self):
        self.ps = coin_weights[self.conf.coin_weights](self.conf)
        samples = np.random.binomial(1,self.ps,size=(self.conf.sample_size, self.conf.reps, self.conf.n))
        print(samples.shape)
        return samples
    

    def run_parallel(self):
        samples = self.get_data()
        self.selection = Selections(conf=self.conf, samples=samples)
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
        self.selection = Selections(conf=self.conf, samples=samples)
        stat_values = [] 
        for i in range(samples.shape[1]):
            result = self.perform_experiment(samples[:,i,:])
            # report the value to show progress
            print(i)
            stat_values.append(result)
        self.stat_values = stat_values
        return stat_values

    def perform_experiment(self, samples):

        initial_samples = samples[:self.conf.initial_size,...]

        ones = np.sum(initial_samples, axis=0)
        zeros = initial_samples.shape[0] - ones

        contingency = np.concatenate((ones[None,...], zeros[None,...]), axis=0)


        contingencies = np.empty((2, self.conf.n, self.conf.sample_size-self.conf.initial_size+1))
        contingencies[:,:,0] = contingency
        for i in range(self.conf.initial_size, self.conf.sample_size):
            coin1, coin2 = self.coin_selection(contingency)
            ops1 = samples[i,coin1]
            ops2 = samples[i,coin2]

            contingency[0, coin1] += ops1
            contingency[0, coin2] += ops2
            contingency[1, coin1] += 1 - ops1
            contingency[1, coin2] += 1 - ops2

            contingencies[:,:,i-self.conf.initial_size+1] = contingency
        return contingencies


    def coin_selection(self, contingency):
        return self.selection.select(contingency)
    
    def save(self):
        makedirs(self.save_path, exist_ok=True)
        selected_coin = np.asarray(self.stat_values)
        name = self.conf.get_sel_name()
        np.save(f"{self.save_path}/contingency_{name}.npy", selected_coin)
