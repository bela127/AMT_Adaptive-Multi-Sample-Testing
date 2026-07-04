from scipy.stats import chi2_contingency, beta
from scipy.special import betaln
import numpy as np

from amt.configuration import Config

class TestStatistic():

    def __init__(self, conf: Config, samples = []) -> None:
        test_mode = conf.test_mode

        self.test_modes = {
            "chi2": chi2_test,
            "beta": beta_dist_test,
            "beta.sum": beta_sum_test,
            "count": count_test,
            "kw": kw_test,
            "mean": mean_test,
            "eval": eval_test
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

def eval_test(contingency, alpha0=1.0, beta0=1.0, alpha1=1.0, beta1=1.0):
    """
    Calculates the Fully Bayesian E-variable for testing coin homogeneity.
    
    Parameters:
    -----------
    contingency : numpy.ndarray
        A 2D array of shape (2, num_coins). 
        contingency[0] contains the count of heads (successes) for each coin.
        contingency[1] contains the count of tails (failures) for each coin.
    alpha0, beta0 : float
        Hyperparameters for the shared Beta prior under H0 (Null).
    alpha1, beta1 : float
        Hyperparameters for the individual Beta priors under H1 (Alternative).
        
    Returns:
    --------
    float
        The computed E-value.
    """
    
    # Ensure input is a numpy array
    contingency = np.asarray(contingency)
    
    # Extract heads (z_x) and tails (N_x - z_x) for each coin
    z_x = contingency[0]
    tails_x = contingency[1]
    
    # ---------------------------------------------------------
    # 1. Calculate log Q (Alternative H1: Many Coins)
    # ---------------------------------------------------------
    # Calculate the log marginal likelihood for each coin individually
    log_q_x = betaln(z_x + alpha1, tails_x + beta1) - betaln(alpha1, beta1)
    
    # Sum the log likelihoods across all coins (equivalent to taking the product)
    log_Q = np.sum(log_q_x)
    
    # ---------------------------------------------------------
    # 2. Calculate log P0 (Null H0: Single Coin)
    # ---------------------------------------------------------
    # Pool all data together
    Z_total = np.sum(z_x)
    Tails_total = np.sum(tails_x)
    
    # Calculate the log marginal likelihood for the pooled data
    log_P0 = betaln(Z_total + alpha0, Tails_total + beta0) - betaln(alpha0, beta0)
    
    # ---------------------------------------------------------
    # 3. Calculate Final E-Variable
    # ---------------------------------------------------------
    # log(E) = log(Q) - log(P0)
    log_E = log_Q - log_P0
    
    # Convert back to standard scale
    e_value = np.exp(log_E)
    
    return e_value

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