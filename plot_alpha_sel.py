from os import makedirs

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


from amt.configuration import Config
from amt.utils import translate_names, create_fig, save_fig, plot_alpha


def post(fig, ax, confs, sig):
    # Get all handles and labels from the plot
    handles, labels = ax.get_legend_handles_labels()

    # Define the number of items for the first legend
    num_items_top = 4  # Change this value to adjust the split

    # Create the first legend at the top
    legend1 = ax.legend(handles[:num_items_top], labels[:num_items_top],
                        loc='upper center', ncol=2)

    # Create the second legend at the bottom
    legend2 = ax.legend(handles[num_items_top:], labels[num_items_top:],
                        loc='lower center', ncol=2)

    # Add the first legend back to the figure.
    # This is necessary because creating the second legend removes the first one.
    ax.add_artist(legend1)


if __name__ == "__main__":
    for n in [5, 10, 15, 20]:#5, 10, 15, 20
        for test_mode in ["mean", "chi2","kw", "beta", "betabinom.comb"]:#"mean", "chi2","kw", "beta", "betabinom.comb"
            confs = []
            for sel_mode in ["beta.med", "equal", "ts.5", "ts", "means", "mean.slow"]:# "ts.5", "ts", "equal", "beta.med", "means", "mean.slow", "beta"
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
                return rf'''Type I error for
{test_name} test at sign. $\alpha={sig}$ for {conf.n} coins'''
            def label(conf: Config):
                test_name, sel_name = translate_names(conf.test_mode, conf.selection_mode)
                return f"{sel_name} selection"
            def save(conf: Config, sig):
                return f"test.mode-{conf.test_mode}_coins-{conf.n}_sig-{sig}"
            
            colors = ['#1b1e3b', '#a1794a', '#16534c', '#447731', '#d484a9', '#c6b4ee']
            plot_alpha(confs, title_func=title, label_func=label, save_func=save, post_func=post, load_path="./test_res/H0_test", save_path = "./alpha_plots/sel.mode", line_colors=colors)