from multiprocessing import Pool, cpu_count
from os import makedirs

import numpy as np
import pandas as pd


from amt.configuration import Config, coin_weights
from amt.selection import Selections



class Experiment():

    def __init__(self, conf: Config, save_path = "./coin_res") -> None:
        self.save_path = save_path
        self.conf = conf
        

    def get_data(self):
        self.ps = coin_weights[self.conf.coin_weights](self.conf)
        samples = np.random.binomial(1,self.ps,size=(self.conf.sample_size, self.conf.reps, self.conf.n))
        print(samples.shape)
        print(self.conf.get_sel_name())
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


class ExperimentBandit():

    def __init__(self, conf: Config, save_path = "./bandit_res") -> None:
        self.save_path = save_path
        self.conf = conf
        

    def get_data(self):
        self.ps = coin_weights[self.conf.coin_weights](self.conf)
        samples = np.random.binomial(1,self.ps,size=(self.conf.sample_size, self.conf.reps, self.conf.n))
        print(samples.shape)
        print(self.conf.get_sel_name())
        return samples
    

    def run_parallel(self):
        samples = self.get_data()
        test_results = []
        contingency_results = []
        with Pool(int(cpu_count())) as pool:
            # call the same function with different data in parallel
            r = 0
            for result in pool.imap_unordered(self.perform_experiment, (samples[:,i,:] for i in range(samples.shape[1]))): #imap_unordered imap
                # report the value to show progress
                print(r)
                contingency_results.append(result[0])
                test_results.append(result[1])
                r+=1
        self.stat_values = (contingency_results, test_results)
        return contingency_results, test_results


    def run(self):
        samples = self.get_data()
        test_results = []
        contingency_results = [] 
        for i in range(samples.shape[1]):
            result = self.perform_experiment(samples[:,i,:])
            # report the value to show progress
            print(i)
            contingency_results.append(result[0])
            test_results.append(result[1])
        self.stat_values = (contingency_results, test_results)
        return contingency_results, test_results

    def perform_experiment(self, samples):
        eps = 0.78
        exploit = 4 #exploitation factor 2 = default; higher = more exploitation; 5 is heavy exploitation
        alpha = 0.05
        r = 0
        t_prev = 0
        candidates = np.ones(shape=(self.conf.n,), dtype=bool)

        initial_samples = samples[:self.conf.initial_size,...]

        ones = np.sum(initial_samples, axis=0)
        zeros = initial_samples.shape[0] - ones

        contingency = np.concatenate((ones[None,...], zeros[None,...]), axis=0)


        contingencies = np.empty((2, self.conf.n, self.conf.sample_size-self.conf.initial_size+1))
        contingencies[:,:,0] = contingency
        tests = np.zeros(shape=(self.conf.sample_size-self.conf.initial_size+1,), dtype=bool)

        i = self.conf.initial_size
        while i<self.conf.sample_size-1:
            r = r +1
            er = 2**(-r)
            t = (2/er**2)*np.log(4*self.conf.n*r**2/alpha)
            if t == np.inf:
                t = np.finfo(np.float64).max
            sample_count = int(t-t_prev)
            
            indices = np.arange(self.conf.n)[candidates]
            for s in range(sample_count):
                indices: np.ndarray[tuple[int, ...], np.dtype[np.signedinteger[np.Any]]] = np.arange(self.conf.n)[candidates]
                np.random.shuffle(indices)
                for coin1 in indices:
                    i=i+1 #we only sample one coin at a time (other algorithms sample two coins in one iteration), which is why we perform double the amount of iterations in the config
                    if i>=self.conf.sample_size-1:
                        break

                    ops1 = samples[i,coin1]

                    contingency[0, coin1] += ops1
                    contingency[1, coin1] += 1 - ops1

                    contingencies[:,:,i-self.conf.initial_size+1] = contingency

                    p_hats = contingency[0,:]/(contingency[0,:]+contingency[1,:])

                    p_hat_max = np.max(p_hats[candidates])
                    p_hat_min = np.min(p_hats[candidates])

                    test = p_hat_max - er*eps > p_hat_min + er*eps
                    tests[i-self.conf.initial_size+1] = test
                if i>=self.conf.sample_size-1:
                    break

            p_hats = contingency[0,:]/(contingency[0,:]+contingency[1,:])

            p_hat_max = np.max(p_hats[candidates])
            p_hat_min = np.min(p_hats[candidates])

            for n in range(self.conf.n):
                if candidates[n]:
                    if p_hats[n] < p_hat_max - er*eps/exploit and p_hats[n] > p_hat_min + er*eps/exploit:
                        candidates[n] = False

        print("coins left:",np.count_nonzero(candidates))
        return contingencies, tests
    
    def save(self):
        makedirs(self.save_path, exist_ok=True)
        contingency_results, test_results = self.stat_values
        selected_coin = np.asarray(contingency_results)
        name = self.conf.get_sel_name()
        np.save(f"{self.save_path}/contingency_{name}.npy", selected_coin)
        tests = np.asarray(test_results)
        name = self.conf.get_test_name()
        np.save(f"{self.save_path}/test_{name}.npy", tests)