from os import makedirs

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


from amt.configuration import Config
from amt.utils import translate_names, create_fig, save_fig, plot_alpha

if __name__ == "__main__":
    for n in [5, 10, 15, 20]:#5, 10, 15, 20
        for sel_mode in ["beta.med", "ts.5", "ts", "equal", "beta", "means", "mean.slow"]:# "ts.5", "ts", "equal", "beta", "means", "mean.slow", "beta.med"
            confs = []
            for test_mode in ["mean", "chi2", "kw", "beta", "betabinom.comb"]:# "mean", "chi2","kw", "beta", "betabinom.comb"
                conf = Config(
                    n = n,
                    m = 0,
                    sample_size = 2000,
                    initial_size = 10,
                    reps = 10000,
                    common_p = 0.5,
                    p_diff = 0.05,
                    selection_mode = sel_mode,
                    test_mode = test_mode,
                    coin_weights = "posdif",
                )
                confs.append(conf)

            def title(conf: Config, sig):
                test_name, sel_name = translate_names(conf.test_mode, conf.selection_mode)
                return rf'''Type I error with
{sel_name} selection at sign. $\alpha={sig}$ for {conf.n} coins'''
            def label(conf: Config):
                test_name, sel_name = translate_names(conf.test_mode, conf.selection_mode)
                return f"{test_name} test"
            def save(conf: Config, sig):
                return f"sel.mode-{conf.selection_mode}_coins-{conf.n}_sig-{sig}"
            
            colors = ['#192d48', '#2b6f39', '#a1794a', '#d490c6', '#c3d9f3']
            plot_alpha(confs, title_func=title, label_func=label, save_func=save, load_path="./test_res/H0_test", save_path = "./alpha_plots/test.mode", line_colors=colors)