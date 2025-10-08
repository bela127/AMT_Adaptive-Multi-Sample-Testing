from os import makedirs

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


from amt.configuration import Config
from amt.utils import translate_names, create_fig, save_fig, plot_power

if __name__ == "__main__":
    for n in [5, 10, 15, 20]:#5, 10, 15, 20
        for test_mode in ["mean", "chi2","kw", "beta", "betabinom.comb"]:#"mean", "chi2","kw", "beta", "betabinom.comb"
            confs = []
            for sel_mode in ["beta.med", "ts", "equal", "mean.slow"]:# "ts.5", "ts", "equal", "beta.med", "means", "mean.slow", "beta"
                
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
                return rf"{test_name} test at $\alpha={sig}$ and {conf.n} coins"
            def label(conf: Config):
                test_name, sel_name = translate_names(conf.test_mode, conf.selection_mode)
                return f"{sel_name} selection"
            def save(conf: Config, sig):
                return f"test.mode-{conf.test_mode}_coins-{conf.n}_sig-{sig}"
            
            def post(fig, ax, confs, sig):
                plt.ylim(0,1)
                plt.legend(ncol=2)
            
            colors = ['#1b1e3b', '#16534c', '#447731', '#a1794a', '#d484a9', '#c6b4ee', '#cbe8f0']
            colors = ['#1b1e3b', '#16534c', '#447731', '#a1794a', '#d484a9', '#c6b4ee']
            colors = ['#1b1e3b', '#447731', '#a1794a', '#c6b4ee']
            plot_power(confs, title_func=title, label_func=label, save_func=save, post_func=post, load_path="./test_res", save_path = "./power_plots/sel.mode", line_colors=colors)