from amt.configuration import Config

import numpy as np

def chernoff_infinite_bound(contingency, conf: Config):
    """
    Type of Bound: Infinite-Horizon Chernoff bound.
    Data Nature:   Binary / Bernoulli [0, 1] (Inherently bounded probability streams).
    Derived From:  Chernoff, H. (1952) / Multiplicative concentration principles 
                   adapted to infinite-time sequential tracking via a Kaufmann & Garivier (2017)
                   style continuous log-log LIL time-uniform boundary.
    
    Description:
    An anytime-valid Chernoff concentration bound operating entirely without a predefined 
    horizon constraint. It replaces the loose polynomial series-peeling structure with a 
    tight asymptotic iterated logarithm envelope. This preserves full Type-I error safety 
    across an infinite timeline while allowing the asymmetric exponential tail optimization 
    to contract efficiently.
    """
    alpha = conf.significance[0]
    num_arms = contingency.shape[1]
    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    # Handle zero-pull divisions safely for the empirical means
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    # Replicate the test's time-uniform log-log LIL penalty structure
    log_term = np.log(num_arms / alpha) + 1.1 * np.log(np.log(np.maximum(2.0, safe_pulls)))
    
    # Asymmetric exponential tail calculation
    main_term = np.sqrt((2.0 * np.minimum(coin_mean, 1.0 - coin_mean) * log_term) / safe_pulls)
    scale_penalty = (0.667 * log_term) / safe_pulls
    radius = main_term + scale_penalty
    
    # Compute active boundaries
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    # Force unpulled arms to complete uncertainty extremes
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def chernoff_horizon_bound(contingency, conf: Config):
    """
    Type of Bound: Horizon-Bounded Chernoff bound.
    Data Nature:   Binary / Bernoulli [0, 1] (Inherently bounded probability streams).
    Derived From:  Chernoff, H. (1952). "A measure of asymptotic efficiency for tests of a 
                   hypothesis based on the sum of observations." Adapted to sequential tracking 
                   via Audibert & Bubeck (2010) horizon-doubling budget allocation.
    
    Description:
    An anytime-valid boundary utilizing Chernoff's multiplicative inequality under a 
    predefined global sample horizon budget (T). It provides tighter exponential 
    concentration tails than Hoeffding by adapting its geometric expansion to the location 
    of the empirical mean. 
    """
    alpha = conf.significance[0]
    T = conf.sample_size * 2
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    # Handle zero-pull divisions safely for empirical means
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    # Horizon-bounded log-peeling log term
    log_term = np.log((num_arms * np.log2(T)) / alpha)
    
    # Asymmetric variance-dependent Chernoff radius
    main_term = np.sqrt((2.0 * np.minimum(coin_mean, 1.0 - coin_mean) * log_term) / safe_pulls)
    scale_penalty = (0.667 * log_term) / safe_pulls
    radius = main_term + scale_penalty
    
    # Compute active boundaries
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    # Force unpulled arms to complete uncertainty extremes
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def hoeffding_variance_horizon_bound(contingency, conf: Config):
    """
    Type of Bound: Horizon-Bounded Hoeffding-style sub-Gaussian bound.
    Data Nature:   Normal (Given Variance) -- generalized from Bernoulli.
    Derived From:  Audibert, J. Y., & Bubeck, S. (2010). "Best arm identification in 
                   multi-armed bandits." (For the log2(T) epoch-doubling horizon structure).
                   Pairs with sub-Gaussian scale derivations from Jamieson et al. (2014).
    
    Description:
    An anytime-valid boundary designed for situations where a maximum global sample 
    budget horizon (T) is predefined. By capping the tracking timeline at T, it avoids 
    the aggressive continuous time penalties of standard anytime bounds (like t^4). 
    Instead, it uses a highly compressed log-multiplier based on the number of maximum 
    possible time-doublings (log2(T)).
    """
    alpha = conf.significance[0]
    variance = 0.25
    T = conf.sample_size * 2
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    log_term = np.log((num_arms * np.log2(T)) / alpha)
    radius = np.sqrt((2.0 * variance * log_term) / safe_pulls)
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def hoeffding_empiric_horizon_bound(contingency, conf: Config):
    """
    Type of Bound: Empirical Hoeffding bound with a global horizon ceiling (UCB-Tuned style).
    Data Nature:   Empiric (The scale parameters adapt dynamically based on sample tracking).
    Derived From:  Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002) "UCB-Tuned" / Audibert et al. (2009)
                   adapted to an Audibert & Bubeck horizon doubling ceiling.
    """
    alpha = conf.significance[0]
    T = conf.sample_size * 2
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    v_hats = coin_mean * (1.0 - coin_mean)
    v_hats = np.maximum(v_hats, 1e-6)
    
    log_term = np.log((num_arms * np.log2(T)) / alpha)
    
    variance_cushion = np.sqrt((2.0 * log_term) / safe_pulls)
    tuned_variance = v_hats + variance_cushion
    
    radius = np.sqrt((2.0 * tuned_variance * log_term) / safe_pulls)
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def hoeffding_variance_infinite_bound(contingency, conf: Config):
    """
    Type of Bound: Hoeffding-style sub-Gaussian continuous bound.
    Data Nature:   Normal (Given Variance) -- generalized from Bernoulli.
    Derived From:  Kalyanakrishnan, S., Tewari, A., Auer, P., & Stone, P. (2012). 
                   Adapted via a Kaufmann & Garivier (2017) style continuous log-log LIL envelope.
    """
    alpha = conf.significance[0]
    variance = 0.25 
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    log_term = np.log(num_arms / alpha) + 1.1 * np.log(np.log(np.maximum(2.0, safe_pulls)))
    
    radius = np.sqrt((2.0 * variance * log_term) / safe_pulls)
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def hoeffding_empiric_infinite_bound(contingency, conf: Config):
    """
    Type of Bound: Empirical Hoeffding bound (Variance-adaptive Sub-Gaussian scale).
    Data Nature:   Empiric (The scale parameters adapt dynamically based on sample tracking).
    Derived From:  Auer et al. (2002) "UCB-Tuned" / Audibert et al. (2009) 
                   paired with a Kaufmann & Garivier (2017) continuous log-log LIL envelope.
    """
    alpha = conf.significance[0]
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    v_hats = coin_mean * (1.0 - coin_mean)
    v_hats = np.maximum(v_hats, 1e-6)
    
    log_term = np.log(num_arms / alpha) + 1.1 * np.log(np.log(np.maximum(2.0, safe_pulls)))
    
    variance_cushion = np.sqrt((2.0 * log_term) / safe_pulls)
    tuned_variance = v_hats + variance_cushion
    
    radius = np.sqrt((2.0 * tuned_variance * log_term) / safe_pulls)
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def bernstein_variance_infinite_bound(contingency, conf: Config):
    """
    Type of Bound: Infinite-Horizon Analytical Bernstein bound.
    Data Nature:   Normal (Given Variance) -- variance parameter is known beforehand.
    Derived From:  Bernstein, S. (1924) adapted to anytime sequential tracking via 
                   a Kaufmann & Garivier (2017) style continuous log-log LIL envelope.
    """
    alpha = conf.significance[0]
    variance = 0.25  
    b_scale = 1.0    
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    log_term = np.log(num_arms / alpha) + 1.1 * np.log(np.log(np.maximum(2.0, safe_pulls)))
    
    radius = np.sqrt((2.0 * variance * log_term) / safe_pulls) + ((2.0 * b_scale * log_term) / (3.0 * safe_pulls))
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def bernstein_empiric_infinite_bound(contingency, conf: Config):
    """
    Type of Bound: Infinite-Horizon Empirical Bernstein bound.
    Data Nature:   Empiric (Variance is dynamically estimated from data stream).
    Derived From:  Maurer, A., & Pontil, M. (2009). "Empirical Bernstein bounds and sample variance."
                   Adapted to infinite-time sequential frameworks via a Kaufmann & Garivier (2017)
                   style continuous log-log LIL time-uniform envelope.
    
    Description:
    An anytime-valid empirical Bernstein bound that operates entirely without a pre-defined 
    horizon ceiling T. It monitors the actual sample variance (v_hats) dynamically. It replaces 
    the loose polynomial series-peeling structure with a tight asymptotic iterated logarithm envelope,
    preserving full Type-I error safety across an infinite timeline while allowing both the 
    variance component and higher-order correction penalty to contract cleanly.
    """
    alpha = conf.significance[0]
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    v_hats = coin_mean * (1.0 - coin_mean)
    v_hats = np.maximum(v_hats, 1e-6)
    
    log_term = np.log(num_arms / alpha) + 1.1 * np.log(np.log(np.maximum(2.0, safe_pulls)))
    
    main_term = np.sqrt((2.0 * v_hats * log_term) / safe_pulls)
    scale_penalty = (0.667 * log_term) / safe_pulls
    radius = main_term + scale_penalty
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def bernstein_variance_horizon_bound(contingency, conf: Config):
    """
    Type of Bound: Analytical Bernstein bound with a global horizon ceiling.
    Data Nature:   Bounded [0, 1] variables with a known variance parameter beforehand.
    Derived From:  Bernstein, S. (1924) / Audibert & Bubeck (2010). 
    
    Description:
    An anytime-valid boundary utilizing Bernstein's inequality with a pre-specified 
    variance parameter and support scale b. By factoring in the structural boundaries 
    of the variable (b_scale), it separates variance constraints from structural scale metrics, 
    yielding a tighter exploration profile for low-variance distributions compared to Hoeffding.
    """
    alpha = conf.significance[0]
    variance = 0.25  
    b_scale = 1.0    
    T = conf.sample_size * 2
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    log_term = np.log((num_arms * np.log2(T)) / alpha)
    
    radius = np.sqrt((2.0 * variance * log_term) / safe_pulls) + ((2.0 * b_scale * log_term) / (3.0 * safe_pulls))
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def bernstein_empiric_horizon_bound(contingency, conf: Config):
    """
    Type of Bound: Empirical Bernstein bound.
    Data Nature:   Empiric (Variance is dynamically estimated from data stream).
    Derived From:  Maurer, A., & Pontil, M. (2009). "Empirical Bernstein bounds and samplevariance."
                   Adapted to bandits by Gabillon et al. (2012) / Audibert et al. (2009).
    
    Description:
    Unlike Hoeffding bounds which assume worst-case variance across the entire domain, 
    this bound monitors sample variance dynamically. It uses a horizon ceiling T 
    combined with a log-log union bound across time. The second term in the radius calculation 
    acts as an additive higher-order correction penalty to protect the bound when sample 
    sizes are low and the empirical variance estimate is highly volatile.
    """
    alpha = conf.significance[0]
    T = conf.sample_size * 2
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    v_hats = coin_mean * (1.0 - coin_mean)
    v_hats = np.maximum(v_hats, 1e-6)
    
    log_term = np.log((num_arms * np.log2(T)) / alpha)
    scale_penalty = (0.667 * log_term) / safe_pulls
    radius = np.sqrt((2.0 * v_hats * log_term) / safe_pulls) + scale_penalty
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def lil_variance_bound(contingency, conf: Config):
    """
    Type of Bound: Law of the Iterated Logarithm (LIL) sub-Gaussian bound.
    Data Nature:   Normal (Given Variance) -- variance parameter is known beforehand.
    Derived From:  Jamieson, K., Malloy, M., Nowak, R., & Bubeck, S. (2014). 
                   "lil' UCB on the empirical complexity of best-arm identification." 
                   Proceedings of the 27th International Conference on Learning Theory (COLT).
    
    Description:
    An anytime-valid boundary that eliminates loose global union-bound multipliers 
    by scaling with an iterated logarithm (log(log(u))) over each arm's individual 
    sample history. Because it assumes a known sub-Gaussian variance parameter, 
    it avoids the higher-order additive estimation penalty, making it exceptionally 
    tight in the early-to-mid sampling phases.
    """
    alpha = conf.significance[0]
    variance = 0.25
    num_arms = contingency.shape[1]
    
    eps_lil = 0.01
    delta_lil = 0.05
    
    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    inner_log = np.log(np.maximum((1.0 + delta_lil) * safe_pulls, 1.01)) + 1.0
    log_term = np.log(inner_log / alpha)
    
    radius = (1.0 + eps_lil) * np.sqrt(((2.0 * variance * (1.0 + delta_lil)) / safe_pulls) * log_term)
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def lil_empiric_bound(contingency, conf: Config):
    """
    Type of Bound: Empirical Bernstein Law of the Iterated Logarithm (LIL) bound.
    Data Nature:   Empiric (Variance is dynamically estimated from data stream).
    Derived From:  Howard, S. R., Ramdas, A., McAuliffe, J. K., & Sekhon, J. S. (2021). 
                   "Time-uniform, non-asymptotic, nonparametric confidence sequences." 
                   The Annals of Statistics. (Adapting empirical variance to LIL boundaries).
    
    Description:
    The ultimate theoretical scaling option for asynchronous sequential tracking. 
    It combines the variance-adaptivity of an Empirical Bernstein bound with the 
    anytime sample-count tightness of the Law of the Iterated Logarithm. 
    """
    alpha = conf.significance[0]
    num_arms = contingency.shape[1]
    
    eps_lil = 0.01
    delta_lil = 0.05
    
    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    v_hats = coin_mean * (1.0 - coin_mean)
    v_hats = np.maximum(v_hats, 1e-6)
    
    inner_log = np.log(np.maximum((1.0 + delta_lil) * safe_pulls, 1.01)) + 1.0
    log_term = np.log(inner_log / alpha)
    
    radius = (1.0 + eps_lil) * (
        np.sqrt(((2.0 * v_hats * (1.0 + delta_lil)) / safe_pulls) * log_term) 
        + (3.0 * log_term / safe_pulls)
    )
    
    U = np.clip(coin_mean + radius, 0.0, 1.0)
    L = np.clip(coin_mean - radius, 0.0, 1.0)
    
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L

def kl_horizon_bound(contingency, conf: Config):
    """
    Type of Bound: Horizon-Bounded Kullback-Leibler (KL) Divergence bound.
    Data Nature:   Binary / Bernoulli [0, 1] (Information-theoretic probability profiles).
    Derived From:  Garivier, A., & Cappé, O. (2011). "The KL-UCB Algorithm for Bounded 
                   Stochastic Bandits." paired with Audibert & Bubeck (2010) horizon ceiling.
    """
    alpha = conf.significance[0]
    T = conf.sample_size * 2  
    num_arms = contingency.shape[1]
    
    pulls_per_arm = contingency[0, :] + contingency[1, :]
    
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    log_term = np.log((num_arms * np.log2(T)) / alpha)
    rhs = log_term / safe_pulls  
    
    U = _solve_kl_boundary(coin_mean, rhs, upper=True)
    L = _solve_kl_boundary(coin_mean, rhs, upper=False)
    
    # Enforce unpulled arm extremes matching your test logic
    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0
    
    return coin_mean, U, L


def kl_infinite_bound(contingency, conf: Config):
    """
    Type of Bound: Infinite-Horizon Kullback-Leibler (KL) Divergence bound.
    Data Nature:   Binary / Bernoulli [0, 1] (Information-theoretic probability profiles).
    Derived From:  Kaufmann, E., & Garivier, A. (2017). "Learning rates for the kl-UCB algorithm."
    """
    alpha = conf.significance[0]
    num_arms = contingency.shape[1]

    pulls_per_arm = contingency[0, :] + contingency[1, :]
    current_t = np.sum(pulls_per_arm)
        
    coin_mean = contingency[0, :] / np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    safe_pulls = np.where(pulls_per_arm == 0, 1, pulls_per_arm)
    
    log_term = np.log(num_arms / alpha) + 3.0 * np.log(np.log(np.maximum(2, current_t)))
    rhs = log_term / safe_pulls
    
    U = _solve_kl_boundary(coin_mean, rhs, upper=True)
    L = _solve_kl_boundary(coin_mean, rhs, upper=False)

    U[pulls_per_arm == 0] = 1.0
    L[pulls_per_arm == 0] = 0.0

    return coin_mean, U, L


def _solve_kl_boundary(p, rhs, upper=True, max_iter=5, eps=1e-7):
    """
    Vectorized hybrid Newton-Raphson / Bisection solver 
    to find the asymmetric KL root q such that d(p, q) = rhs.
    """
    p = np.clip(p, eps, 1.0 - eps)
    
    if upper:
        active_low = p.copy()
        active_high = np.ones_like(p) * (1.0 - eps)
    else:
        active_low = np.ones_like(p) * eps
        active_high = p.copy()
        
    if upper:
        q = np.minimum(1.0 - eps, p + np.sqrt(rhs / 2.0))
    else:
        q = np.maximum(eps, p - np.sqrt(rhs / 2.0))

    with np.errstate(divide='ignore', invalid='ignore'):
        for _ in range(max_iter):
            q = np.clip(q, active_low + eps, active_high - eps)
            
            f = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q)) - rhs
            
            if upper:
                active_high = np.where(f > 0, q, active_high)
                active_low = np.where(f <= 0, q, active_low)
            else:
                active_low = np.where(f > 0, q, active_low)
                active_high = np.where(f <= 0, q, active_high)
            
            df = (-p / q) + ((1.0 - p) / (1.0 - q))
            df = np.where(np.abs(df) < eps, np.sign(df) * eps, df)
            
            q_new = q - f / df
            invalid = (q_new <= active_low) | (q_new >= active_high) | np.isnan(q_new)
            q = np.where(invalid, (active_low + active_high) / 2.0, q_new)
        
    return q





bandit_bounds = {
    "chernoff.infinite": chernoff_infinite_bound,
    "chernoff.horizon": chernoff_horizon_bound,
    "hoeffding.variance.horizon": hoeffding_variance_horizon_bound,
    "hoeffding.empiric.horizon": hoeffding_empiric_horizon_bound,
    "hoeffding.variance.infinite": hoeffding_variance_infinite_bound,
    "hoeffding.empiric.infinite": hoeffding_empiric_infinite_bound,
    "bernstein.variance.infinite": bernstein_variance_infinite_bound,
    "bernstein.empiric.infinite": bernstein_empiric_infinite_bound,
    "bernstein.variance.horizon": bernstein_variance_horizon_bound,
    "bernstein.empiric.horizon": bernstein_empiric_horizon_bound,
    "lil.variance": lil_variance_bound,
    "lil.empiric": lil_empiric_bound,
    "kl.horizon": kl_horizon_bound,
    "kl.infinite": kl_infinite_bound,
}