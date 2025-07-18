from scipy.stats import betabinom
import numpy as np

from atm.configuration import Config

class MultiTests():

    def __init__(self, conf: Config, samples = []) -> None:
        self.alpha = np.asarray(conf.significance)[:,None,None,None]
        test_mode = conf.test_mode

        self.test_modes = {
            "betabinom.crit": self.betabinom_crit,
            "betabinom.pval": self.betabinom_pval,
            "betabinom.pmf": self.betabinom_pmf,
            "betabinom.comb": self.betabinom_comb,
        }

        self.test = self.test_modes[test_mode]

    def betabinom_crit(self, contingency):
        comb_cont = np.sum(contingency, axis=1)[:,None]
        null_cont = (comb_cont-contingency)

        coin_sum = np.sum(contingency, axis=0)

        null_con1 = null_cont[0]
        null_con2 = null_cont[1]

        fbb = betabinom(coin_sum, null_con1+1, null_con2+1)

        corr_alpha = self.alpha/contingency.shape[1] # 1 - (1 - alpha)**(1/n) #alpha/n

        crit_low = fbb.ppf(corr_alpha/2)
        crit_high = fbb.ppf(1-corr_alpha/2)

        heads = contingency[0]

        tests = np.logical_or(heads < crit_low, heads > crit_high)
        return tests
    
    def betabinom_pval(self, contingency):
        comb_cont = np.sum(contingency, axis=1)[:,None]
        null_cont = (comb_cont-contingency)

        coin_sum = np.sum(contingency, axis=0)

        null_con1 = null_cont[0]
        null_con2 = null_cont[1]

        fbb = betabinom(coin_sum, null_con1+1, null_con2+1)

        corr_alpha = self.alpha/contingency.shape[1] # 1 - (1 - alpha)**(1/n) #alpha/n

        count = contingency[0]

        pval_left = fbb.cdf(count)

        tests = np.logical_or(pval_left < corr_alpha/2, pval_left > 1-(corr_alpha/2))
        return tests

    def betabinom_pmf(self, contingency):
        comb_cont = np.sum(contingency, axis=1)[:,None]
        null_cont = (comb_cont-contingency)

        coin_sum = np.sum(contingency, axis=0)

        null_con1 = null_cont[0]
        null_con2 = null_cont[1]

        fbb = betabinom(coin_sum, null_con1+1, null_con2+1)

        corr_alpha = self.alpha/contingency.shape[1] # 1 - (1 - alpha)**(1/n) #alpha/n

        heads = contingency[0]

        # --- P-value calculation using PMF method (Vectorized/Pseudo-Vectorized) ---

        # 1. Calculate PMF at observed heads for all observations
        # This is already vectorized by SciPy
        pmf_at_observed_h = fbb.pmf(heads)
        # pmf_at_observed_h now has shape (5, 50, 20)

        # 2. Determine the maximum possible 'n' across all observations
        # This sets the size of our 'virtual' arange.
        max_coin_sum = np.max(coin_sum) # max_coin_sum is a scalar

        # 3. Create a numpy array representing all possible 'k' values up to max_coin_sum
        # This array will serve as our 'all_possible_heads_for_current_n'
        # Shape: (max_coin_sum + 1,)
        k_values = np.arange(max_coin_sum + 1)[:, None, None, None]# Reshape k_values for broadcasting

        # 4. Calculate PMF for ALL possible 'k' values, for EACH of your (5,50,20) distributions.
        # This requires broadcasting 'k_values' against your distribution parameters.
        # We need to reshape k_values to (max_coin_sum + 1, 1, 1, 1) to broadcast correctly
        # against coin_sum, null_con1, null_con2 which are (5,50,20).
        # The output 'all_pmfs_broadcasted' will have shape (max_coin_sum + 1, 5, 50, 20)
        all_pmfs_broadcasted = fbb.pmf(k_values)

        # 5. Mask out invalid PMF values for each observation
        # Where 'k' (row index) > 'n' (coin_sum), the PMF should effectively be 0.
        # We can use k_values and coin_sum for this.
        # Create a mask that is True where k <= n for each distribution
        # Shape: (max_coin_sum + 1, 5, 50, 20)
        valid_k_mask = k_values <= coin_sum

        # Apply the mask: set invalid PMF values to 0
        all_pmfs_broadcasted_masked = np.where(valid_k_mask, all_pmfs_broadcasted, 0.0)


        # 6. Sum PMFs that are <= pmf_at_observed_h for each distribution
        # This comparison must also broadcast.
        # The comparison is between:
        #   - all_pmfs_broadcasted_masked (shape max_coin_sum+1, 5,50,20)
        #   - pmf_at_observed_h (shape 5,50,20) - needs to be broadcasted to (1, 5,50,20)
        # The result of the comparison will be boolean, (max_coin_sum+1, 5,50,20)
        is_less_than_or_equal = (all_pmfs_broadcasted_masked <= pmf_at_observed_h)

        # Sum along the 'k_values' axis (axis=0)
        p_values_two_sided = np.sum(is_less_than_or_equal * all_pmfs_broadcasted_masked, axis=0)

        tests = p_values_two_sided < corr_alpha

        return tests
    
    def betabinom_comb(self, contingency):
        coin_sum = np.sum(contingency, axis=0)
        max_coin_sum = np.max(coin_sum)
        if max_coin_sum > 500:
            return self.betabinom_pval(contingency)
        else:
            return self.betabinom_pmf(contingency)