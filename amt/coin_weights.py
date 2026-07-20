import numpy as np
from amt.configuration import Config

def calc_coin_weights_posdif(conf: Config):
    same_ps = np.ones(shape=(conf.n,))*conf.common_p
    all_diff_ps = np.ones(shape=(conf.n,))*conf.common_p+conf.p_diff
    m_diff_ps = np.concatenate((same_ps[:conf.n-conf.m], all_diff_ps[:conf.m]))

    ps = m_diff_ps
    return ps

def calc_coin_weights_simdif(conf: Config):
    pos_diff_count = conf.m//2 + conf.m%2
    neg_diff_count = conf.m//2 

    p_diff = conf.p_diff/2 if neg_diff_count > 0 else conf.p_diff

    same_ps = np.ones(shape=(conf.n-conf.m,))*conf.common_p
    pos_diff_ps = np.ones(shape=(pos_diff_count,))*conf.common_p + p_diff
    neg_diff_ps = np.ones(shape=(neg_diff_count,))*conf.common_p - p_diff



    m_diff_ps = np.concatenate((neg_diff_ps,same_ps, pos_diff_ps))
    return m_diff_ps

coin_weights = {
        "posdif": calc_coin_weights_posdif,
        "simdif": calc_coin_weights_simdif
    }