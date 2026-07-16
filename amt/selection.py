from scipy.stats import chi2_contingency, beta
from scipy.special import betaln
import numpy as np
from amt.configuration import Config
from amt.bandit_bounds import bandit_bounds

class Selections():

    def __init__(self, conf: Config, samples = None) -> None:
        self.m = conf.m
        self.n: int = conf.n
        self.sample_size = conf.sample_size
        self.alpha_c = conf.alpha_c

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
                selection_mode = "ts"

        
        self.selection_modes = {
            "evar": self.coin_selection_e_variable,
            "adapt": self.coin_selection_adapt,
            "rand": self.coin_selection_random,
            "equal": self.coin_selection_equal,#
            "cons": self.coin_selection_concentrate,
            "opt": self.coin_selection_optimal,
            "adapt.slow": self.coin_selection_adapt_equal,
            "adapt.cons": self.coin_selection_adapt_cons,
            "beta": self.coin_selection_beta,#
            "ts": self.coin_selection_ts,
            "beta.rand": self.coin_selection_beta_rand,
            "beta.null": self.coin_selection_beta_null,
            "beta.inf": self.coin_selection_beta_inf,
            "beta.inf2": self.coin_selection_beta_inf2,
            "beta.med": self.coin_selection_beta_med,
            "beta.max": self.coin_selection_beta_max,
            "means": self.coin_selection_mean,
            "mean.slow": self.coin_selection_mean_equal
        }

        if sel_params[0] == "bandit":
            selection_mode = "bandit"
            self.selection_modes["bandit"] = self.get_bandit_selection(sel_params[1], conf)
        
        self.select = self.selection_modes[selection_mode]

    def get_bandit_selection(self, selection_mode, conf: Config):
        bandit_selection_modes = {
            "max": coin_selection_by_bound_max,
            "ratio": coin_selection_by_bound_ratio
        }

        bandit_selection = bandit_selection_modes[selection_mode]
        bandit_bound = bandit_bounds[conf.bandit_kind]

        def selection_function(contingency):
            return bandit_selection(contingency, lambda cont: bandit_bound(cont, conf))

        return selection_function



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


    def coin_selection_e_variable(self, contingency, alpha1=1.0, beta1=1.0):
        """
        Selection mechanism based on the localized e variables.
        This adaptively targets the streams maximizing the alternative hypothesis likelihood,
        simplifying the theoretical proofs for Growth Rate Optimality.
        """
        z_arms = contingency[0, :]
        tails_arms = contingency[1, :]
        
        z_total = np.sum(z_arms)
        tails_total = np.sum(tails_arms)
        
        z_rest = z_total - z_arms
        tails_rest = tails_total - tails_arms
        
        # Calculate the log marginal likelihoods for the partitioned alternative hypotheses
        log_q_s = betaln(z_arms + alpha1, tails_arms + beta1) - betaln(alpha1, beta1)
        log_q_rest = betaln(z_rest + alpha1, tails_rest + beta1) - betaln(alpha1, beta1)
        
        # Total localized alternative log likelihood
        log_Q_matrix = log_q_s + log_q_rest
        
        # The global null is identical for all arms, meaning maximizing the alternative 
        # directly maximizes the local e variable. We extract the indices of the highest values.
        sorted_indices = np.argsort(log_Q_matrix)
        
        # Return the two most divergent streams for subsequent sampling
        coin1 = sorted_indices[-1]
        coin2 = sorted_indices[-2]
        
        return coin1, coin2

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

        crit1 = fr.ppf(self.alpha_c/2)
        crit2 = fr.ppf(1-self.alpha_c/2)

        p_low = beta.cdf(crit1, *(contingency+1))
        p_high = 1-beta.cdf(crit2, *(contingency+1))

        coin1 = np.argmax(p_low)
        coin2 = np.argmax(p_high)

        return coin1, coin2

    def coin_selection_beta_max(self, contingency):

        fr = beta(*(contingency+1))

        crit_low = fr.ppf(0.2)
        crit_high = fr.ppf(0.8)

        coin1 = np.argmin(crit_low)
        coin2 = np.argmax(crit_high)

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

                if p <= current_best: # type: ignore
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

def coin_selection_by_bound_ratio(contingency, bound_function):
    """
    Type: Decoupled Functional Interval Ratio Selection (True Asymmetric Beta_Med).
    Goal: Maximize interval penetration relative to the true asymmetric width,
          anchored around the true empirical median to balance targeted divergence 
          optimization with uncertainty extraction.
    """
    # 1. Execute the unified bound function to pull exact parameters
    coin_mean, U, L = bound_function(contingency)
    
    coin_sum = np.sum(contingency, axis=0)
    
    # 2. Calculate the true width of the asymmetric intervals
    width = U - L
    width = np.where(width == 0, 1e-8, width) # Protect against zero-width limits
    
    # 3. CORRECTED: Establish baseline anchor using the true empirical means
    median_anchor = np.median(coin_mean)
    
    # 4. Dynamic fence margin scaled as half the average interval width
    delta_margin = 0.5 * np.mean(width)
    crit_low = median_anchor - delta_margin
    crit_high = median_anchor + delta_margin
    
    # Force unpulled arms to maximum uncertainty space to maintain valid exploration
    U_eff = np.where(coin_sum == 0, 1.0, U)
    L_eff = np.where(coin_sum == 0, 0.0, L)
    width_eff = np.where(coin_sum == 0, 1.0, width)
    
    # 5. Compute penetration efficiency ratios
    ratio_low = (crit_low - L_eff) / width_eff
    ratio_high = (U_eff - crit_high) / width_eff
    
    coin1 = np.argmax(ratio_low)
    coin2 = np.argmax(ratio_high)
    
    # 6. Single-arm boundary dominance tracking (Overlap protection)
    if coin1 == coin2:
        sort_idx = np.argsort(ratio_high)
        coin2 = sort_idx[-2] if sort_idx[-1] == coin1 else sort_idx[-1]
        
    return coin1, coin2

def coin_selection_by_bound_max(contingency, bound_function):
    """
    Type: Decoupled Functional Extreme-Boundary Selection (Standard Bandit Baseline).
    Goal: Find the stream maximizing the upper bound potential and the stream 
          minimizing the lower bound potential independently in O(N) time.
    """
    # 1. Execute the unified bound function to extract raw asymmetric horizons
    # Expected return shape for each: np.ndarray of shape (N,)
    _, U, L = bound_function(contingency)
    
    coin_sum = np.sum(contingency, axis=0)
    
    # 2. Force unpulled arms to maximum uncertainty extremes to seed initial data
    U_eff = np.where(coin_sum == 0, 1.0, U)
    L_eff = np.where(coin_sum == 0, 0.0, L)
    
    # 3. Standard Bandit Targeting: Maximize Upper Bound and Minimize Lower Bound
    # coin1 tracks the lowest lower bound (chasing lower divergence or high uncertainty)
    coin1 = np.argmin(L_eff)
    # coin2 tracks the highest upper bound (chasing upper divergence or high uncertainty)
    coin2 = np.argmax(U_eff)
    
    # 4. Single-Arm Overlap Protection
    # If the same unobserved or massive-variance arm occupies both absolute extremes,
    # route the second allocation slot to the runner-up upper bound to maintain a valid pair.
    if coin1 == coin2:
        sort_idx = np.argsort(U_eff)
        coin2 = sort_idx[-2] if sort_idx[-1] == coin1 else sort_idx[-1]
        
    return coin1, coin2