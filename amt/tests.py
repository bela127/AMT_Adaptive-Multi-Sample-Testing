from typing import Optional

from scipy.stats import beta
from scipy.stats import chi2_contingency
from scipy.special import betaln
from scipy.stats import betabinom
from scipy.stats import chi2
import numpy as np

from amt.configuration import Config
from amt.bandit_bounds import bandit_bounds

class Test():

    def __init__(self, conf: Optional[Config] = None) -> None:
        
        # Complete collection matrix using "." as the key separator
        self.test_modes = {
            "chernoff.horizon":            get_bandit_test("chernoff.horizon"),
            "chernoff.infinite":           get_bandit_test("chernoff.infinite"),

            "hoeffding.variance.horizon":  get_bandit_test("hoeffding.variance.horizon"),
            "hoeffding.variance.infinite": get_bandit_test("hoeffding.variance.infinite"),
            "hoeffding.empiric.horizon": get_bandit_test("hoeffding.empiric.horizon"),
            "hoeffding.empiric.infinite": get_bandit_test("hoeffding.empiric.infinite"),

            "bernstein.variance.horizon":  get_bandit_test("bernstein.variance.horizon"),
            "bernstein.variance.infinite": get_bandit_test("bernstein.variance.infinite"),
            "bernstein.empiric.horizon":  get_bandit_test("bernstein.empiric.horizon"),
            "bernstein.empiric.infinite": get_bandit_test("bernstein.empiric.infinite"),

            "lil.variance":              get_bandit_test("lil.variance"),
            "lil.empiric":               get_bandit_test("lil.empiric"),

            "kl.horizon":                  get_bandit_test("kl.horizon"),  # Computational expensive, but well founded
            "kl.infinite":                 get_bandit_test("kl.infinite"), # Computational expensive, but well founded

            # --- ASYMPTOTIC LIKELIHOOD RATIO ---
               "glrt.horizon":                glrt_horizon_test,
               "glrt.infinite":               glrt_infinite_test,
            
            # --- E-VARIABLES & SAFE TESTING ---
            "bayesian.e.variable":         bayesian_e_variable, #Not very powerful, but a good baseline
              "one.vs.rest.beta.mixture":    one_vs_rest_beta_mixture_test, # Our new e-var approach
              "beta.mixture.infinite":       beta_mixture_infinite_test,
              "ebp.mixture.infinite":        ebp_mixture_infinite_test,
            "betting.e.variable":          self.betting_evariable_test,

            # --- NON ANYTIME ADAPTIVE TESTING ---
               "bayesian.beta":               bayesian_beta, #Not very powerful and computational expensive
                   "bayesian.beta.loo":           bayesian_beta_loo_test, # Unstable (alpha inflation) at low sample sizes, but powerful at high sample sizes
                                                                    #Increased Bonnferroni from 4 to num arms, noticeable improvement in stability, but still not perfect.
                                                                    #Loss in power -> better use bayesian_beta

            "betabinom.pmf":               betabinom_pmf_test,
               "betabinom.extreme.pmf":       betabinom_extreme_pmf_test,
                    "betabinom.isolated.extreme":  betabinom_isolated_extreme_test, # computational very expensive and equivalent to "betabinom.pmf".

            # --- CLASIC TESTING ---

            "chi2": chi2_test,
            "kw": kw_test
        
        }

        if conf is None:
            return
        
        test_mode = conf.test_mode
        self.test = self.test_modes[test_mode]
        
        ################### beting e-variable stateful tracking for anytime stopping ###################
        
        self.num_streams = conf.n if conf else 1
        
        # Track single global log-wealth process.
        # Starts at log(1.0) = 0.0
        self.log_global_e = 0.0  
        
        # Initialize running historical counts with Laplace priors
        self.history_z = np.ones(self.num_streams)      
        self.history_tails = np.ones(self.num_streams)  
        
        # Track previous contingency to calculate sequential chunk deltas
        self.prev_contingency = np.zeros((2, self.num_streams))

    def betting_evariable_test(self, contingency, conf: Config):
        """
        Type of Bound: E-Variable / Test by Betting (Martingale-Based Growth).
        Methodology:   BET Algorithm (Binary Evaluation by Betting / Betting E-Testing).
        Data Nature:   Binary / Bernoulli [0, 1] Sequential Multi-Stream Data.
        Derived From:  Shafer, G. (2021) "Testing by Betting" / 
                        Waudby-Smith & Ramdas (2023) "Estimating Means by Betting" /
                        Grünwald, de Heide, & Koolen (2024).
        
        Description:
        An anytime-valid testing framework implementing the BET paradigm, which transforms 
        sequential multi-stream heterogeneity testing into an active asset-growth game against 
        the null hypothesis ($H_0$). Instead of tracking streams independently, this method 
        maintains a single global, non-negative capital process in log-space to ensure robust 
        martingale preservation and eliminate numerical stability issues.

        At each sequential interval, the algorithm evaluates each stream against a strictly 
        predictable Leave-One-Out (LOO) baseline derived from all other concurrent streams. 
        Using a game-theoretic Kelly-optimal allocation constrained to prevent ruin, betting 
        fractions are formulated for each stream. The performance multipliers of these parallel 
        bets are combined via an arithmetic mean at each step, ensuring that the unified global 
        wealth process behaves as a valid martingale under the null. If the alternative hypothesis 
        is true, the global wealth grows exponentially, triggering a rejection the moment it 
        breaches the $1/\alpha$ threshold.
        """
        alpha_target = conf.significance[0]
        log_e_threshold = np.log(1.0 / alpha_target)

        z_x = contingency[0, :]
        tails_x = contingency[1, :]

        # 1. Extract delta block
        delta_z = z_x - self.prev_contingency[0, :]
        delta_tails = tails_x - self.prev_contingency[1, :]
        stream_deltas = delta_z + delta_tails
        total_delta = np.sum(stream_deltas)

        # Short-circuit evaluation if no new data arrived anywhere
        if total_delta == 0:
            return self.log_global_e >= log_e_threshold

        # 2. Predictable historical estimates (per-stream)
        history_n = self.history_z + self.history_tails
        p_hat_stream = self.history_z / history_n

        # 3. Vectorized Leave-One-Out (LOO) Null Baseline
        total_history_z = np.sum(self.history_z)
        total_history_n = np.sum(history_n)

        loo_history_z = total_history_z - self.history_z
        loo_history_n = total_history_n - history_n

        p_hat_null = loo_history_z / loo_history_n
        p_hat_null = np.clip(p_hat_null, 1e-5, 1 - 1e-5)

        # 4. Vectorized Kelly fraction calculation
        g = (p_hat_stream - p_hat_null) / (p_hat_null * (1.0 - p_hat_null))
        
        max_g = 1.0 / (1.0 - p_hat_null) - 1e-4
        min_g = -1.0 / p_hat_null + 1e-4
        g = np.clip(g, min_g, max_g)
        
        # 5. Compute linear multipliers per stream for this specific block
        mult_success = 1.0 + g * (1.0 - p_hat_null)
        mult_failure = 1.0 - g * p_hat_null
        
        # Compute the compounding growth factor for this chunk's samples
        # If a stream has 0 updates, its stream_multipliers will be exactly 1.0
        stream_multipliers = (mult_success ** delta_z) * (mult_failure ** delta_tails)
        
        
        # Under H0, the arithmetic mean of these concurrent multipliers is strictly a valid martingale step.
        block_multiplier = np.mean(stream_multipliers)
        
        # Safety check to prevent log(0) in catastrophic deviations
        block_multiplier = max(block_multiplier, 1e-15)

        # 7. Accumulate into global wealth process
        self.log_global_e += np.log(block_multiplier)

        # 8. Update running counts for the next iteration
        self.history_z += delta_z
        self.history_tails += delta_tails
        self.prev_contingency = contingency.copy()

        return self.log_global_e >= log_e_threshold


###### anytime stopping tests #######
def get_bandit_test(bandit_kind: str):
    """
    Higher-Order Factory to generate unified anytime-valid stopping rules
    for boundary-driven sequential frameworks.
    """
    def bandit_test(contingency, conf: Config) -> bool:
        num_arms = contingency.shape[1]
        pulls_per_arm = contingency[0, :] + contingency[1, :]
        current_t = np.sum(pulls_per_arm)
        
        # 1. Warm-up guardrail: Prevent premature evaluation
        if current_t < 2 * num_arms:
            return False
            
        # 2. Extract boundaries using the decoupled parameter signature
        coin_mean, U, L = bandit_bounds[bandit_kind](contingency, conf)
        
        # 3. Create an active mask to isolate pulled arms
        valid_mask = pulls_per_arm > 0
        if np.sum(valid_mask) < 2:
            return False
            
        # 4. Standard Omnibus Crossing Check: Leader Lower Bound > Challenger Upper Bound
        return np.max(L[valid_mask]) > np.min(U[valid_mask])
    
    return bandit_test


#### logistic regression tests #######

def glrt_infinite_test(contingency, conf: Config):
    """
    Type of Bound: Infinite-Horizon Generalized Likelihood Ratio Test (GLRT).
    Data Nature:   Normal Asymptotic (Time-Uniform Continuous Scaling).
    Derived From:  Kaufmann, E., & Garivier, A. (2017). "Learning rates for the kl-UCB algorithm"
                   / Robbins, H. (1970) mixture boundaries for likelihood ratios.
    
    Description:
    An asymptotic tracking test built on a self-normalized log-likelihood ratio 
    operating safely over an infinite tracking timeline. It drops the hard horizon 
    requirement by scaling the permissible divergence barrier dynamically with the 
    total accumulated observations across the entire system.
    """
    beta_param = conf.significance[0]
    variance = 0.25
    num_arms = contingency.shape[1]
    
    pulls_per_arm = contingency[0, :] + contingency[1, :]
    current_t = np.sum(pulls_per_arm)
    if current_t < 2 * num_arms: 
        return False
        
    p_hats = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    threshold = np.log(num_arms / beta_param) + 3.0 * np.log(np.log(np.maximum(2, current_t)))
    
    # Fully vectorized all-pairs distance and sample pooling matrices
    # Shape: (num_arms, num_arms)
    mean_diffs = p_hats[:, None] - p_hats[None, :]
    
    n_i = pulls_per_arm[:, None]
    n_j = pulls_per_arm[None, :]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pooled_allocations = (n_i * n_j) / (n_i + n_j)
        all_pairs_stats = pooled_allocations * (mean_diffs**2) / (2.0 * variance)
        
    # Mask out self-comparison diagonals and unpulled arm pairs
    np.fill_diagonal(all_pairs_stats, 0.0)
    mask = (pulls_per_arm == 0)[:, None] | (pulls_per_arm == 0)[None, :]
    all_pairs_stats[mask] = 0.0
    
    # Multi-sample criteria: Reject H0 if ANY pair breaks the log-barrier
    return np.max(all_pairs_stats) >= threshold


def glrt_horizon_test(contingency, conf: Config):
    """
    Type of Bound: Generalized Likelihood Ratio Test (GLRT) Bound.
    Data Nature:   Normal Asymptotic (Generalized via plug-in Gaussian parameters).
    Derived From:  Garivier, A., & Kaufmann, E. (2016). "Optimal Best Arm Identification 
                   with Fixed Confidence." Ann. Statist.
    
    Description:
    An asymptotic tracking test built on the self-normalized log-likelihood ratio. 
    Instead of checking individual independent boundaries, it computes a global 
    profile likelihood ratio between the current empirical best arm and all competing 
    arms, calibrating the stop threshold against a log-horizon budget multiplier.
    """
    beta_param = conf.significance[0]
    variance = 0.25  
    T = conf.sample_size * 2  # Fixed maximum horizon budget across the sequence
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    current_t = np.sum(pulls_per_arm)
    if current_t < 2 * num_arms:
        return False

    p_hats = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    #Finite-sample sub-Gaussian tracking constant (+0.5) to prevent late-horizon boundary breaches
    threshold = np.log((num_arms * np.log2(T)) / beta_param) + 0.5
    
    # Fully vectorized all-pairs distance and sample pooling matrices
    mean_diffs = p_hats[:, None] - p_hats[None, :]
    
    n_i = pulls_per_arm[:, None]
    n_j = pulls_per_arm[None, :]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pooled_allocations = (n_i * n_j) / (n_i + n_j)
        all_pairs_stats = pooled_allocations * (mean_diffs**2) / (2.0 * variance)
        
    np.fill_diagonal(all_pairs_stats, 0.0)
    mask = (pulls_per_arm == 0)[:, None] | (pulls_per_arm == 0)[None, :]
    all_pairs_stats[mask] = 0.0
    
    return np.max(all_pairs_stats) >= threshold


###### anytime alpha stopping and continuation #######

def one_vs_rest_beta_mixture_test(contingency: np.ndarray, conf: Config):
    """
    Type of Bound: One-vs-Rest Beta-Binomial Likelihood Ratio E-Variable (Partitioned Likelihood Ratio).
    Data Nature:   Binary / Bernoulli [0, 1] (Exact Likelihood, Non-Asymptotic).
    Derived From:  Howard et al. (2021) / Grünwald, de Heide, & Koolen (2024).
    
    Description:
    An anytime-valid multi-sample testing framework that partitions the alternative 
    hypothesis into a sequence of isolated low-dimensional spaces. It evaluates each 
    individual stream against the collective pooled data of all remaining streams, 
    aggregating the resulting spatial E-variables into a single global arithmetic mean.
    """
    alpha_target = conf.significance[0]
    e_threshold = 1.0 / alpha_target

    # Uniform prior hyperparameters
    alpha0, beta0 = 1.0, 1.0
    alpha1, beta1 = 1.0, 1.0

    # contingency shape: (2, num_arms)
    z_arms = contingency[0, :]
    tails_arms = contingency[1, :]
    pulls_per_arm = z_arms + tails_arms
    num_arms = len(z_arms)
    
    total_pulls = np.sum(pulls_per_arm)
    # Minimum sample gating across the entire system
    if total_pulls < 2 * num_arms:
        return False, 0.0
        
    z_total = np.sum(z_arms)
    tails_total = np.sum(tails_arms)
    
    # 1. H0: Global homogeneity across all samples (Scalar value)
    log_P0_global = betaln(z_total + alpha0, tails_total + beta0) - betaln(alpha0, beta0)

    # 2. H1: Isolate each arm s and calculate its complement "the rest"
    # Vectorized subtraction yields arrays of shape (num_arms,)
    z_rest = z_total - z_arms
    tails_rest = tails_total - tails_arms
    
    # 3. Compute marginal likelihoods across all arms in parallel
    log_q_s = betaln(z_arms + alpha1, tails_arms + beta1) - betaln(alpha1, beta1)
    log_q_rest = betaln(z_rest + alpha1, tails_rest + beta1) - betaln(alpha1, beta1)
    
    # Total H1 log-likelihood per arm assignment
    log_Q_matrix = log_q_s + log_q_rest
    
    # 4. Calculate individual E-variables safely
    with np.errstate(divide='ignore', invalid='ignore'):
        e_variables = np.exp(log_Q_matrix - log_P0_global)
        
    # Mask out structurally invalid states (e.g., an arm or its rest has no samples)
    invalid_mask = (pulls_per_arm == 0) | (pulls_per_arm == total_pulls)
    e_variables[invalid_mask] = 0.0
    
    # 5. Take the arithmetic mean to create a valid, highly powerful global E-variable
    global_e_variable = np.mean(e_variables)
    
    return global_e_variable >= e_threshold

def bayesian_e_variable(contingency, conf: Config):
    """
    Type of Bound: Global Beta-Binomial Likelihood Ratio E-Variable (Fully Decoupled Likelihood Ratio).
    Data Nature:   Binary / Bernoulli [0, 1] (Exact Likelihood, Non-Asymptotic).
    Derived From:  Howard et al. (2021) / Grünwald, de Heide, & Koolen (2024).
    
    Description:
    An anytime-valid global testing framework for multi-stream binary data. It evaluates 
    a single joint likelihood ratio comparing a completely pooled global null hypothesis 
    (1 parameter) against a fully decoupled alternative hypothesis where every individual 
    stream is modeled with its own independent parameter.
    """
    alpha_target = conf.significance[0]
    e_threshold = 1.0 / alpha_target
    
    # Matching uniform prior hyperparameters
    alpha0, beta0 = 1.0, 1.0
    alpha1, beta1 = 1.0, 1.0

    z_x = contingency[0, :]
    tails_x = contingency[1, :]
    
    # Log Q (Alternative H1 - independent parameters per stream)
    log_q_x = betaln(z_x + alpha1, tails_x + beta1) - betaln(alpha1, beta1)
    log_Q = np.sum(log_q_x)
    
    # Log P0 (Null H0 - single pooled global parameter)
    Z_total = np.sum(z_x)
    Tails_total = np.sum(tails_x)
    log_P0 = betaln(Z_total + alpha0, Tails_total + beta0) - betaln(alpha0, beta0)
    
    current_e = np.exp(log_Q - log_P0)
    return current_e >= e_threshold


def ebp_mixture_infinite_test(contingency, conf: Config):
    """
    Type of Bound: Exponential Bounded Process Continuous Normal Mixture E-Variable (All-Pairs Sub-Gaussian Bound).
    Data Nature:   Continuous / Sub-Gaussian Frequencies (All-Pairs Multi-Sample Check).
    Derived From:  Howard, S. R., Ramdas, A., McAuliffe, J. K., & Sekhon, J. S. (2021).
    
    Description:
    An anytime-valid multi-sample testing framework designed for sub-Gaussian sequential 
    processes. It utilizes a continuous normal mixture martingale transformation to evaluate 
    pairwise mean differences across an intrinsic scaling variance process, applying a spatial 
    union bound correction layer over all arm combinations.
    """
    alpha_target = conf.significance[0]
    sigma_squared = 0.25  
    
    # Rho represents the prior variance hyperparameter of the mixture.
    rho_tuning = 1.0  
    num_arms = contingency.shape[1]
    
    # Pairwise union bound correction layer
    num_pairs = max(1, (num_arms * (num_arms - 1)) // 2)
    e_threshold = (1.0 / alpha_target) * num_pairs

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    current_t = np.sum(pulls_per_arm)
    if current_t < 2 * num_arms:
        return False
        
    p_hats = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    # Fully vectorized all-pairs calculation
    mean_diffs = p_hats[:, None] - p_hats[None, :]
    n_i = pulls_per_arm[:, None]
    n_j = pulls_per_arm[None, :]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        n_effective = (n_i * n_j) / (n_i + n_j)
        
        # The intrinsic variance process V_t for a normalized 
        # sub-Gaussian sequence scales directly as n_effective.
        v_process = n_effective
        
        # S_t^2 mapped directly inside the exponential numerator
        s_squared = (n_effective ** 2) * (mean_diffs ** 2) / sigma_squared
        
        # Standard Howard/Robbins normal mixture martingale transformation
        scaling_factor = np.sqrt(rho_tuning / (v_process + rho_tuning))
        exponent = s_squared / (2.0 * (v_process + rho_tuning))
        
        e_matrix = scaling_factor * np.exp(exponent)
        
    # Mask diagonals and unpulled tracking pairs
    np.fill_diagonal(e_matrix, 0.0)
    mask = (pulls_per_arm == 0)[:, None] | (pulls_per_arm == 0)[None, :]
    e_matrix[mask] = 0.0
    
    return np.max(e_matrix) >= e_threshold

def beta_mixture_infinite_test(contingency, conf):
    """
    Type of Bound: Exact Pairwise Beta-Binomial Likelihood Ratio E-Variable (All-Pairs Likelihood Ratio).
    Data Nature:   Binary / Bernoulli [0, 1] (Exact Likelihood, Non-Asymptotic).
    Derived From:  Howard et al. (2021) / Grünwald, de Heide, & Koolen (2024).
    
    Description:
    An anytime-valid multi-sample testing framework that decomposes a multi-arm system 
    into all possible localized pairs. It computes exact independent Beta-Binomial likelihood 
    ratios for each distinct pair of streams and applies a spatial family-wise union bound 
    correction layer to the localized thresholds.
    """
    alpha_target = conf.significance[0]
    num_arms = contingency.shape[1]
    
    # Pairwise union bound correction layer
    num_pairs = max(1, (num_arms * (num_arms - 1)) // 2)
    e_threshold = (1.0 / alpha_target) * num_pairs

    # Uniform prior hyperparameters
    alpha0, beta0 = 1.0, 1.0
    alpha1, beta1 = 1.0, 1.0

    z_arms = contingency[0, :]
    tails_arms = contingency[1, :]
    pulls_per_arm = z_arms + tails_arms
    
    # Minimum sample gating across the entire system
    if np.sum(pulls_per_arm) < 2 * num_arms:
        return False
        
    # 1. Compute H1 Marginal Likelihoods for all individual arms
    # Vector of length (num_arms,)
    log_q_individual = betaln(z_arms + alpha1, tails_arms + beta1) - betaln(alpha1, beta1)
    
    # Convert to 2D pairwise combinations via broadcasting: log_Q(i, j) = log_q(i) + log_q(j)
    log_Q_matrix = log_q_individual[:, None] + log_q_individual[None, :]
    
    # 2. Compute H0 Marginal Likelihoods for all pooled arm pairs
    # Create pairwise pooled matrices using broadcasting
    z_pooled = z_arms[:, None] + z_arms[None, :]
    tails_pooled = tails_arms[:, None] + tails_arms[None, :]
    
    log_P0_matrix = betaln(z_pooled + alpha0, tails_pooled + beta0) - betaln(alpha0, beta0)
    
    # 3. Calculate E-Variable Matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        e_matrix = np.exp(log_Q_matrix - log_P0_matrix)
        
    # Mask diagonals (comparing an arm to itself is invalid)
    np.fill_diagonal(e_matrix, 0.0)
    
    # Mask pairs where either arm hasn't been pulled yet
    mask = (pulls_per_arm == 0)[:, None] | (pulls_per_arm == 0)[None, :]
    e_matrix[mask] = 0.0
    
    # Return True if the maximum pairwise E-variable breaches the union-bound threshold
    return np.max(e_matrix) >= e_threshold


###### Credible Interval Tests #######

def bayesian_beta(contingency, conf: Config):
    """
    Type of Bound: Bayesian Credible Interval (Quantile Concentration with Bonferroni Correction).
    Data Nature:   Binary (Explicitly locked to the conjugate Beta-Binomial framework).
    Derived From:  Foundational Bayesian Inference / Probability Theory paired with 
                   the Bonferroni Inequality for Family-Wise Error Rate (FWER) control.
    """
    beta_param = conf.significance[0]
    num_arms = contingency.shape[1]
    
    # Update posterior parameters directly from the contingency entries
    alphas = 1.0 + contingency[0, :]
    betas_attr = 1.0 + contingency[1, :]

    corr_alpha = beta_param / num_arms #(num_arms*2)
    
    lower_bounds = beta.ppf(corr_alpha, alphas, betas_attr)
    upper_bounds = beta.ppf(1.0 - corr_alpha, alphas, betas_attr)
    
    # Find the critical boundary frontiers directly from the interval matrices
    h_double_star = np.argmax(lower_bounds)
    l_double_star = np.argmin(upper_bounds)
    
    # Stop if the highest lower boundary strictly clears the lowest upper boundary
    return lower_bounds[h_double_star] > upper_bounds[l_double_star]


def bayesian_beta_loo_test(contingency, conf):
    """
        Type of Bound: Bayesian Reference Interval (Fixed Horizon / Post-Hoc Test).
        Methodology:   Bayesian Leave-One-Out (LOO) Credible Reference Testing.
        Data Nature:   Binary / Bernoulli [0, 1] Aggregate Multi-Arm Contingency Data.
        Derived From:  Standard Bayesian Beta-Binomial Conjugacy / 
                        Order Statistics and Extreme Value Boundary Estimations.
        
        Description:
        For evaluation, the algorithm calculates a Family-Wise Error Rate (FWER) corrected 
        credible interval for each individual arm. It isolates the empirical leader and laggard 
        arms, dynamically comparing their respective posterior boundaries against an 
        vectorized Leave-One-Out (LOO) background distribution built from all remaining arms. A 
        rejection is triggered if the leader's lower performance boundary strictly clears the 
        background's upper threshold, or if the laggard's upper boundary drops below the 
        background's lower threshold.
        """
    beta_param = conf.significance[0]
    num_arms = contingency.shape[1]
    
    # Extract views to avoid repetitive 2D indexing overhead
    successes = contingency[0, :]
    failures = contingency[1, :]
    pulls_per_arm = successes + failures
    
    # Prevents extreme alpha inflation caused by loose order statistics at low sample sizes.
    # Rule of thumb: require an average of at least 30-50 samples per arm before testing.
    MIN_PULLS_PER_ARM = 30 
    if np.any(pulls_per_arm < MIN_PULLS_PER_ARM):
        return False
        
    # STRICT FWER MULTI-ARM CORRECTION
    corr_beta = beta_param / num_arms
    
    # STABLE JEFFREYS INVARIANT PRIOR
    prior_mass = 0.5
    
    alphas_init = prior_mass + successes
    betas_init = prior_mass + failures
    
    # Vectorized computation of bounds across all arms
    lower_bounds = beta.ppf(corr_beta, alphas_init, betas_init)
    upper_bounds = beta.ppf(1.0 - corr_beta, alphas_init, betas_init)
    
    leader_idx = np.argmax(lower_bounds)
    laggard_idx = np.argmin(upper_bounds)
    
    if leader_idx == laggard_idx:
        return False

    total_successes = successes.sum()
    total_failures = failures.sum()
    total_pulls = total_successes + total_failures
    
    # --- Check Leader ---
    bg_pulls_leader = total_pulls - pulls_per_arm[leader_idx]
    
    # Background also uses the Jeffreys prior
    alpha_prior_leader = (total_successes - successes[leader_idx]) + prior_mass
    beta_prior_leader = (total_failures - failures[leader_idx]) + prior_mass
    
    bg_upper_threshold = beta.ppf(1.0 - corr_beta, alpha_prior_leader, beta_prior_leader)
    if lower_bounds[leader_idx] > bg_upper_threshold:
        return True
        
    # --- Check Laggard ---
    bg_pulls_laggard = total_pulls - pulls_per_arm[laggard_idx]
    
    alpha_prior_laggard = (total_successes - successes[laggard_idx]) + prior_mass
    beta_prior_laggard = (total_failures - failures[laggard_idx]) + prior_mass
    
    bg_lower_threshold = beta.ppf(corr_beta, alpha_prior_laggard, beta_prior_laggard)
    if upper_bounds[laggard_idx] < bg_lower_threshold:
        return True
                
    return False


def betabinom_pmf_test(contingency, conf: Config):
    """
    Type of Bound: Exact Vectorized Non-Parametric Beta-Binomial PMF Test.
    Data Nature:   Count-Based (Integer counts of successes/failures instead of frequencies).
    Derived From:  Leave-One-Out Predictive Distribution under a Dirichlet-Multinomial/Beta-Binomial
                   conjugate framework. Pairwise combination via multi-sample reduction.
    """
    alpha_target = conf.significance[0]
    num_arms = contingency.shape[1]
    pulls_per_arm = np.sum(contingency, axis=0)
    if np.any(pulls_per_arm == 0): return False
        
    corr_alpha = alpha_target / num_arms
    
    coin_sum = pulls_per_arm
    null_con1 = np.sum(contingency[0, :]) - contingency[0, :]
    null_con2 = np.sum(contingency[1, :]) - contingency[1, :]

    fbb = betabinom(coin_sum, null_con1 + conf.prior[0], null_con2 + conf.prior[1])
    
    k_values = np.arange(np.max(coin_sum) + 1)[:, None]
    all_pmfs = fbb.pmf(k_values) # pyright: ignore[reportAttributeAccessIssue]
    all_pmfs_masked = np.where(k_values <= coin_sum, all_pmfs, 0.0)
    
    pmf_at_obs = fbb.pmf(contingency[0, :]) # pyright: ignore[reportAttributeAccessIssue]
    p_values = np.sum((all_pmfs_masked <= pmf_at_obs) * all_pmfs_masked, axis=0)
    
    return np.any(p_values < corr_alpha)

def betabinom_extreme_pmf_test(contingency, conf: Config):
    """
    Type of Bound: Exact Two-Sided Direct Pairwise Beta-Binomial Bound Test.
    Data Nature:   Count-Based (Integer counts).
    Derived From:  Bidirectional Posterior Predictive Framework mapping leader 
                   and laggard mutually against each other's isolated intervals.
    """
    alpha_target = conf.significance[0]
    num_arms = contingency.shape[1]
    
    pulls_per_arm = contingency[0, :] + contingency[1, :]
    if np.any(pulls_per_arm == 0):
        return False
        
    # Full combinatorial pair-selection envelope correction because we isolate 
    # a single pair directly from the pool without background pooling.
    num_pairs = max(1, (num_arms * (num_arms - 1)) // 2)
    corr_alpha = alpha_target / num_pairs

    # 1. Continuous Beta intervals used strictly as a fast frontier selection filter
    alphas_selection = 1.0 + contingency[0, :]
    betas_selection = 1.0 + contingency[1, :]
    
    lower_bounds = beta.ppf(corr_alpha, alphas_selection, betas_selection)
    upper_bounds = beta.ppf(1.0 - corr_alpha, alphas_selection, betas_selection)
    
    leader_idx = np.argmax(lower_bounds)
    laggard_idx = np.argmin(upper_bounds)
    
    if leader_idx == laggard_idx:
        return False

    # Extract isolated extreme properties
    x_leader = contingency[0, leader_idx]
    n_leader = pulls_per_arm[leader_idx]
    
    x_laggard = contingency[0, laggard_idx]
    n_laggard = pulls_per_arm[laggard_idx]
    
    # ==========================================
    # PERSPECTIVE 1: Test Leader against Laggard
    # ==========================================
    alpha_prior_l = 1.0 + x_laggard
    beta_prior_l = 1.0 + (n_laggard - x_laggard)
    
    fbb_leader = betabinom(n_leader, alpha_prior_l, beta_prior_l)
    leader_upper_bound = fbb_leader.ppf(1.0 - corr_alpha)
    
    if x_leader > leader_upper_bound:
        return True

    # ==========================================
    # PERSPECTIVE 2: Test Laggard against Leader
    # ==========================================
    alpha_prior_r = 1.0 + x_leader
    beta_prior_r = 1.0 + (n_leader - x_leader)
    
    fbb_laggard = betabinom(n_laggard, alpha_prior_r, beta_prior_r)
    laggard_lower_bound = fbb_laggard.ppf(corr_alpha)
    
    if x_laggard < laggard_lower_bound:
        return True
            
    return False


def betabinom_isolated_extreme_test(contingency, conf: Config):
    """
    Type of Bound: Exact Predictive Beta-Binomial Bound-Overlap Test.
    Data Nature:   Count-Based (Integer counts).
    Derived From:  Bayesian Predictive Quantile Intervals with Bonferroni Correction.
                   Direct discrete analog to the continuous bayesian_beta test.
    """
    alpha_target = conf.significance[0]
    num_arms = contingency.shape[1]
    
    pulls_per_arm = contingency[0, :] + contingency[1, :]
    if np.any(pulls_per_arm == 0):
        return False
        
    # Correct for K arms and 2-sided tail bounds (Matches bayesian_beta exactly)
    corr_alpha = alpha_target / (2.0 * num_arms)
    
    # Total counts across the entire system
    total_successes = np.sum(contingency[0, :])
    total_failures = np.sum(contingency[1, :])
    
    # Isolate the background pool by subtracting the current arm's data
    bg_successes = total_successes - contingency[0]
    bg_failures = total_failures - contingency[1]

    # Build the background prior parameters (Beta-conjugate)
    alpha_prior = 1.0 + bg_successes
    beta_prior = 1.0 + bg_failures
    
    fbb = betabinom(pulls_per_arm, alpha_prior, beta_prior)
    
    # Calculate the exact lower and upper integer success thresholds
    lower_count_bounds = fbb.ppf(corr_alpha)
    upper_count_bounds = fbb.ppf(1.0 - corr_alpha)
        
    # The actual observed success counts we are testing
    observed_successes = contingency[0]
    
    # An arm triggers a stopping condition if its actual observed success count 
    # breaks completely outside the predicted background bounds.
    # (e.g., a leader's successes clear the upper bound, or a laggard's drop below the lower bound)
    significant_high = observed_successes > upper_count_bounds
    significant_low = observed_successes < lower_count_bounds
    
    return np.any(significant_high) or np.any(significant_low)


def chi2_test(contingency, conf: Config):
    alpha_target = conf.significance[0]

    try:
        res = chi2_contingency(contingency)
        p_val = res[1]
        return p_val < alpha_target
    except ValueError:
        print(f"missing data -> test can not be rejected")
        return False


def kw_test(contingency, conf: Config):
    alpha_target = conf.significance[0]

    try:
        # 1. Row/Col summations
        coin_sum = np.sum(contingency, axis=0)
        low_high_sum = np.sum(contingency, axis=1)
        total_sum = np.sum(contingency, axis=(0, 1))

        # Check for zero totals or empty tables to avoid division by zero
        if total_sum <= 1 or np.any(coin_sum == 0):
            return False

        # 2. Optimized ranks calculation
        mid_low = (1 + low_high_sum[0]) / 2
        mid_high = (low_high_sum[0] + 1 + total_sum) / 2

        average_ranks = (mid_low * contingency[0, :] + mid_high * contingency[1, :]) / coin_sum

        expected_average_rank = (total_sum + 1) / 2
        expected_rank_variance = (total_sum**2 - 1) / 12

        inner_metric = (average_ranks - expected_average_rank)**2 * coin_sum / expected_rank_variance

        H = (total_sum - 1) / total_sum * np.sum(inner_metric)

        # 3. Tie Correction
        tie_correction = 1.0 - (low_high_sum[0]**3 - low_high_sum[0] + low_high_sum[1]**3 - low_high_sum[1]) / (total_sum**3 - total_sum)
        
        # Guard against zero-division if all values belong to a single tier
        if tie_correction == 0:
            return False
            
        h = H / tie_correction

        # 4. Degree of Freedom & P-Value Calculation
        # In Kruskal-Wallis, df = (number of groups) - 1. 
        # Here, each coin/column represents a group.
        num_groups = contingency.shape[1]
        df = num_groups - 1

        # Use the public chi2.sf (Survival Function) to get the right-tailed p-value
        p_value = chi2.sf(h, df)

        return p_value < alpha_target

    except (ValueError, ZeroDivisionError):
        # Gracefully handle situations where math fails due to empty/uninitialized data
        return False