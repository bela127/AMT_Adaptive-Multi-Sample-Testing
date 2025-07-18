from scipy.stats import chi2_contingency, beta
import numpy as np

from atm.configuration import Config

class Tests():

    def __init__(self, conf: Config, samples = []) -> None:
        test_mode = conf.test_mode

        self.test_modes = {
            "chi2": chi2_test,
            "beta": beta_dist_test,
            "beta.sum2": beta_sum_test,
            "count": count_test,
            "kw": kw_test,
            "mean": mean_test
        }

        self.test = self.test_modes[test_mode]


def chi2_test(contingency):
    try:
        res = chi2_contingency(contingency)
        stat = res[0]
    except ValueError:
        stat = 0
        print(f"missing data -> test rejected")
    return stat

def beta_dist_test(contingency):

    comb_cont = np.sum(contingency, axis=1)[:,None]
    crit1 = beta.ppf(0.025, *(comb_cont-contingency+1))
    crit2 = beta.ppf(0.975, *(comb_cont-contingency+1))

    p_low = beta.cdf(crit1, *(contingency+1))
    p_high = 1-beta.cdf(crit2, *(contingency+1))

    p_max = np.maximum(p_low, p_high)

    return np.max(p_max)

def beta_sum_test(contingency):

    comb_cont = np.sum(contingency, axis=1)[:,None]
    comb_cont_per_coin = comb_cont-contingency

    crit1 = beta.ppf(0.025, *(comb_cont_per_coin+1))
    crit2 = beta.ppf(0.975, *(comb_cont_per_coin+1))

    p_low = beta.cdf(crit1, *(contingency+1))
    p_high = 1-beta.cdf(crit2, *(contingency+1))

    p_max = np.maximum(p_low, p_high)

    return np.sum(p_max)

def mean_test(contingency):

    comb_cont = np.sum(contingency, axis=1)[:,None]
    null_cont = (comb_cont-contingency)

    null_sum = np.sum(null_cont, axis=0)

    null_average = null_cont[0] / null_sum #empiric p under H0 = of average coin

    coin_sum = np.sum(contingency, axis=0)

    coin_average = contingency[0] / coin_sum #empiric p of coin

    coin_error = (null_average - coin_average)**2

    scaled_error = coin_error * coin_sum # scale error by number of trials per coin, so that larger samples have more weight

    average_error = np.sum(scaled_error, axis=0) / np.sum(coin_sum, axis=0) #divide by total number of trials to get average error
    
    return average_error

def count_test(contingency):

    coin_sum = np.sum(contingency, axis=0)
    total_sum = np.sum(contingency, axis=(0,1))

    coin_percent = coin_sum / total_sum*contingency.shape[1]

    return np.max(coin_percent)

def kw_test(contingency):

    coin_sum = np.sum(contingency, axis=0)
    low_high_sum = np.sum(contingency, axis=1)
    total_sum = np.sum(contingency, axis=(0,1))


    mid_low = (1 + low_high_sum[0])/2
    mid_high = (low_high_sum[0]+ 1 + total_sum)/2

    average_ranks = (mid_low*contingency[0,:] + mid_high*contingency[1,:])/coin_sum

    expected_average_rank = (total_sum + 1)/2
    expected_rank_variance = (total_sum**2-1)/12

    inner_metric = (average_ranks - expected_average_rank)**2 * coin_sum / expected_rank_variance

    H = (total_sum-1)/total_sum * np.sum(inner_metric)

    tie_correction =1- (low_high_sum[0]**3 - low_high_sum[0] + low_high_sum[1]**3 - low_high_sum[1])/(total_sum**3 - total_sum)
    h = H / tie_correction

    return h