from scipy.stats import chi2_contingency, beta
import numpy as np
from amt.configuration import Config

class Selections():

    def __init__(self, conf: Config, samples = None) -> None:
        self.m = conf.m
        self.n = conf.n
        self.sample_size = conf.sample_size
        selection_mode = conf.selection_mode

        sel_params = selection_mode.split(".")

        if selection_mode == "opt":
            if samples is not None:
                self.init_opt(samples)
            else:
                print("You have to call 'init_opt' to use opt selection mode.")

        
        if sel_params[0] == "ts":
            if len(sel_params) == 1:
                self.smoothing = 1
            else:
                self.smoothing = int(sel_params[1])

        
        self.selection_modes = {
            "adapt": self.coin_selection_adapt,
            "rand": self.coin_selection_random,
            "equal": self.coin_selection_equal,#
            "cons": self.coin_selection_concentrate,
            "opt": self.coin_selection_optimal,
            "adapt.slow": self.coin_selection_adapt_equal,
            "adapt.cons": self.coin_selection_adapt_cons,
            "beta": self.coin_selection_beta,#
            "ts": self.coin_selection_ts,
            "ts.5": self.coin_selection_ts,
            "beta.rand": self.coin_selection_beta_rand,
            "beta.null": self.coin_selection_beta_null,
            "beta.inf": self.coin_selection_beta_inf,
            "beta.inf2": self.coin_selection_beta_inf2,
            "beta.med": self.coin_selection_beta_med,
            "means": self.coin_selection_mean,
            "mean.slow": self.coin_selection_mean_equal
        }


        self.select = self.selection_modes[selection_mode]


    def init_opt(self, samples):
        ones = np.sum(samples, axis=(0,1))
        zeros = samples.shape[0]*samples.shape[1] - ones

        contingency = np.concatenate((ones[None,...], zeros[None,...]), axis=0)

        comb_cont = np.sum(contingency, axis=1)[:,None]
        crit1 = beta.ppf(0.2, *(comb_cont-contingency+1))
        crit2 = beta.ppf(0.8, *(comb_cont-contingency+1))

        p_low = beta.cdf(crit1, *(contingency+1))
        p_high = 1-beta.cdf(crit2, *(contingency+1))

        self.opt_coin1 = np.argmax(p_low)
        self.opt_coin2 = np.argmax(p_high)

    def coin_selection_random(self, contingency):
        return np.random.choice(10,size=2,replace=False)


    def coin_selection_optimal(self, contingency):
        if self.m == 0:
            return self.coin_selection_equal(contingency)
        return self.opt_coin1, self.opt_coin2

    def coin_selection_equal(self, contingency):
        obs_nrs = np.sum(contingency, axis=0)
        partitions = np.argpartition(obs_nrs,2)
        return partitions[:2]

    def coin_selection_concentrate(self, contingency):
        obs_nrs = np.sum(contingency, axis=0)
        if np.all(obs_nrs == obs_nrs[0]):
            return self.coin_selection_random(contingency)
        else:
            partitions = np.argpartition(obs_nrs,-2)
            return partitions[-2:]

    def coin_selection_beta(self, contingency):

        comb_cont = np.sum(contingency, axis=1)[:,None]
        crit1 = beta.ppf(0.2, *(comb_cont-contingency+1))
        crit2 = beta.ppf(0.8, *(comb_cont-contingency+1))

        p_low = beta.cdf(crit1, *(contingency+1))
        p_high = 1-beta.cdf(crit2, *(contingency+1))

        coin1 = np.argmax(p_low)
        coin2 = np.argmax(p_high)

        return coin1, coin2
    
    def coin_selection_beta_med(self, contingency):
        coin_sum = np.sum(contingency, axis = 0)
        coin_mean = contingency[0] / coin_sum
        total_sum = np.sum(coin_sum, axis=0)
        aver_count = total_sum / contingency.shape[1]

        median_mean = np.median(coin_mean, axis=0)

        fr = beta(median_mean*aver_count+1, (1 - median_mean)*aver_count+1)

        crit1 = fr.ppf(0.2)
        crit2 = fr.ppf(0.8)

        p_low = beta.cdf(crit1, *(contingency+1))
        p_high = 1-beta.cdf(crit2, *(contingency+1))

        coin1 = np.argmax(p_low)
        coin2 = np.argmax(p_high)

        return coin1, coin2

    def coin_selection_mean(self, contingency):

        coin_sum = np.sum(contingency, axis = 0)
        coin_mean = contingency[0] / coin_sum

        coin1 = np.argmax(coin_mean)
        coin2 = np.argmin(coin_mean)

        return coin1, coin2

    def coin_selection_ts(self, contingency):

        comb_cont = np.sum(contingency, axis=1)[:,None]
        frv = beta(*(comb_cont-contingency+1))

        samples = frv.rvs(size=(self.smoothing, contingency.shape[-1]))
        s_mean = np.mean(samples, axis=0)

        coin1 = np.argmax(s_mean)
        coin2 = np.argmin(s_mean)

        return coin1, coin2



    def coin_selection_beta_null(self, contingency):

        comb_cont = np.sum(contingency, axis=1)[:,None]
        crit1 = beta.ppf(0.2, *(comb_cont-contingency+1))
        crit2 = beta.ppf(0.8, *(comb_cont-contingency+1))

        p_low = beta.cdf(crit1, *(contingency+1))
        p_high = 1-beta.cdf(crit2, *(contingency+1))

        p_diff = np.abs(p_low - p_high)

        coin1 = np.argmax(p_diff)
        coin2 = np.argmin(p_diff)

        return coin1, coin2

    def coin_selection_beta_rand(self, contingency):

        comb_cont = np.sum(contingency, axis=1)[:,None]
        crit1 = beta.ppf(0.025, *(comb_cont-contingency+1))
        crit2 = beta.ppf(0.975, *(comb_cont-contingency+1))

        p_low = beta.cdf(crit1, *(contingency+1))
        p_high = 1-beta.cdf(crit2, *(contingency+1))

        p_low_exp = p_low
        p_high_exp = p_high

        p_low_norm = p_low_exp / np.sum(p_low_exp, dtype=np.float64)
        p_high_norm = p_high_exp / np.sum(p_high_exp, dtype=np.float64)

        coin1 = np.random.choice(np.arange(contingency.shape[1]),p=p_low_norm)
        coin2 = np.random.choice(np.arange(contingency.shape[1]),p=p_high_norm)

        return coin1, coin2

    def coin_selection_beta_inf(self, contingency):

        comb_cont = np.sum(contingency, axis=1)[:,None]
        crit1 = beta.ppf(0.025, *(comb_cont-contingency+1))
        crit2 = beta.ppf(0.975, *(comb_cont-contingency+1))

        p_low = beta.cdf(crit1, *(contingency+1))
        p_high = 1-beta.cdf(crit2, *(contingency+1))

        p_low_exp = - np.log(p_low)
        p_high_exp = - np.log(p_high)

        p_low_norm = p_low_exp / np.sum(p_low_exp, dtype=np.float64)
        p_high_norm = p_high_exp / np.sum(p_high_exp, dtype=np.float64)

        coin1 = np.random.choice(np.arange(contingency.shape[1]),p=p_low_norm)
        coin2 = np.random.choice(np.arange(contingency.shape[1]),p=p_high_norm)

        return coin1, coin2


    def coin_selection_beta_inf2(self, contingency):

        comb_cont = np.sum(contingency, axis=1)[:,None]
        crit1 = beta.ppf(0.025, *(comb_cont-contingency+1))
        crit2 = beta.ppf(0.975, *(comb_cont-contingency+1))

        p_low = beta.cdf(crit1, *(contingency+1))
        p_high = 1-beta.cdf(crit2, *(contingency+1))

        p_low_exp = - np.log(p_low)
        p_high_exp = - np.log(p_high)

        p_low_norm = p_low_exp / np.sum(p_low_exp, dtype=np.float64)
        p_high_norm = p_high_exp / np.sum(p_high_exp, dtype=np.float64)

        coin1 = np.random.choice(np.arange(contingency.shape[1]),p=p_low_norm)
        coin2 = np.random.choice(np.arange(contingency.shape[1]),p=p_high_norm)

        return coin1, coin2

    def coin_selection_adapt(self, contingency):
        current_best = 1
        best_coin1 = self.n-1
        best_coin2 = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                contingency1 = contingency.T[i]
                contingency2 = contingency.T[j]
                part_con = np.concatenate((contingency1[:,None],contingency2[:, None]), axis=1)

                try:
                    res = chi2_contingency(part_con)
                    p = res[1]
                except ValueError:
                    p = 0
                    print(f"missing data -> sampled more")

                if p <= current_best:
                    current_best = p
                    best_coin1 = i
                    best_coin2 = j
        return best_coin1, best_coin2

    def coin_selection_adapt_equal(self, contingency):
        focus = np.sum(contingency) / (self.sample_size*2)
        if np.random.uniform() > focus:
            return self.coin_selection_equal(contingency)
        else:
            return self.coin_selection_adapt(contingency)
    
    def coin_selection_mean_equal(self, contingency):
        focus = np.sum(contingency) / (self.sample_size*2)
        if np.random.uniform() > focus:
            return self.coin_selection_equal(contingency)
        else:
            return self.coin_selection_mean(contingency)
        
    def coin_selection_adapt_cons(self, contingency):
        focus = np.sum(contingency) / (self.sample_size*2)

        if np.random.uniform() > focus:
            return self.coin_selection_adapt(contingency)
        else:
            return self.coin_selection_concentrate(contingency)


