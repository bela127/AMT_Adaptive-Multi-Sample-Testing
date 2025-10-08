from os import makedirs

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


from amt.configuration import Config
from amt.utils import translate_names, create_fig, save_fig, plot_power

if __name__ == "__main__":  
    for test_mode in ["mean", "chi2","kw", "beta", "betabinom.comb"]:#"mean", "chi2","kw", "beta", "betabinom.comb"
        for sel_mode in ["ts.5", "ts", "equal", "beta", "means", "mean.slow", "beta.med"]:# "ts.5", "ts", "equal", "beta", "means", "mean.slow", "beta.med"
            confs = []
            for n in [5, 10, 15, 20]:#5, 10, 15, 20
                conf = Config(
                    n = n,
                    m = 1,
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
                return rf"Power for {test_name} test with selection {sel_name} at $\alpha={sig}$"
            def label(conf: Config):
                return rf"$n = {conf.n}$"
            def save(conf: Config, sig):
                return f"sel.mode-{conf.selection_mode}_test.mode-{conf.test_mode}_sig-{sig}"
            
            colors = ['#163d4e', '#54792f', '#d07e93', '#c1caf3']
            plot_power(confs, title_func=title, label_func=label, save_func=save, load_path="./test_res/H1", save_path = "./power_plots/coins", line_colors=colors)