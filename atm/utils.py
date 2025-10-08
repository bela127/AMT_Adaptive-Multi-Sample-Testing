import os
from os import makedirs

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

from amt.configuration import Config


def translate_keys(test_mode, sel_mode):
    test_translations = {
        "betabinom.comb": "betabinom",
        }
    sel_translations = {
        "mean.slow": "meanslow",
        "ts.5": "ts5",
        "beta.med": "beta",
        "beta": "beta.old",
        }
    return test_translations.get(test_mode, test_mode), sel_translations.get(sel_mode, sel_mode)


def translate_names(test_mode, sel_mode):
    test_translations = {
        "betabinom.comb": "Beta-Binomial",
        "mean": "Mean",
        "chi2": "Chi-Squared",
        "kw": "Kruskal-Wallis",
        "beta": "Beta",
        }
    sel_translations = {
        "mean.slow": "Mean Slow",
        "ts.5": "TS5",
        "ts": "TS",
        "beta.med": "Beta",
        "means": "Means",
        "equal": "Equal",
        "opt": "Oracle",
        }
    return test_translations.get(test_mode, test_mode), sel_translations.get(sel_mode, sel_mode)

def translate_test_colors(test_mode):
    colors5 = ['#192d48', '#2b6f39', '#a1794a', '#d490c6', '#c3d9f3']
    test_translations = {
        "betabinom.comb": colors5[0],
        "mean": colors5[1],
        "chi2": colors5[2],
        "kw": colors5[3],
        "beta": colors5[4],
        }
    return test_translations.get(test_mode, test_mode)


def save_fig(fig, name, path, format="svg"):
    fig.tight_layout()
    loc = os.path.join(path,f"{name}.{format}")
    fig.savefig(loc, format=format, bbox_inches='tight', transparent="True", pad_inches=0)
    fig.clf()

def set_size(width="paper_2c", fraction:float=1, subplots=(1, 1), hfrac=1., vfrac=1.):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'paper_2c':
        width_pt = 252
    elif width == 'paper':
        width_pt = 516
    elif width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif isinstance(width, float):
        width_pt = width
    else:
        raise ValueError(f"{width=} is no known size")

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio: float = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * 1.2 * hfrac * (subplots[0] / subplots[1])

    fig_width_in = fig_width_in * vfrac #Partial width

    return (fig_width_in, fig_height_in)

def create_fig(width="paper_2c", fraction:float =1, subplots=(1, 1), hfrac:float=1, vfrac:float=1):
    plt.style.use('seaborn-v0_8-paper')

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False, #True,
        "text.latex.preamble": r'\usepackage{amssymb}',
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }

    plt.rcParams.update(tex_fonts)


    fig, axs = plt.subplots(subplots[0], subplots[1], figsize=set_size(width=width, fraction=fraction, subplots=subplots, hfrac=hfrac, vfrac=vfrac))
    #axs.set_prop_cycle(line_cycler)
    return fig, axs


def plot_power(confs: list[Config], title_func, label_func, save_func, post_func = None, save_path: str = "./power_plots", load_path = "./test_res", line_styles = [], line_colors = []) -> None:

    conf = confs[0]

    makedirs(save_path, exist_ok=True)
    sizes = np.arange(0,conf.sample_size - conf.initial_size + 1)

    def plot_data(sizes, conf, line_style = None, line_color=None):
        name = conf.get_test_name()
        power = np.load(f"{load_path}/power_{name}.npy")
        print(name)
        rcParams["lines.dashed_pattern"] = (5,10)

        if line_style is not None and line_color is not None:
            plt.plot(sizes, power[i], label=label_func(conf), color=line_color, linestyle=line_style)
        elif line_style is not None:
            plt.plot(sizes, power[i], label=label_func(conf), linestyle=line_style)
        elif line_color is not None:
            plt.plot(sizes, power[i], label=label_func(conf), color=line_color)
        else:
            plt.plot(sizes, power[i], label=label_func(conf))
    
    for i, sig in enumerate(confs[0].significance):
        fig, axs = create_fig(hfrac=0.85)

        if line_styles and line_colors:
            for conf, line_style, line_color in zip(confs, line_styles, line_colors):
                plot_data(sizes, conf, line_style, line_color)
        elif line_styles:
            for conf, line_style in zip(confs, line_styles):
                plot_data(sizes, conf, line_style = line_style)
        elif line_colors:
            for conf, line_color in zip(confs, line_colors):
                plot_data(sizes, conf, line_color = line_color)
        else:
            for conf in confs:
                plot_data(sizes, conf)
            
        plt.title(title_func(conf, sig))
        plt.xlabel("Iterations")
        plt.ylabel("Power")
        plt.ylim(0, 1)
        plt.grid()

        if post_func is not None:
            post_func(fig, axs, confs, sig)
        else:
            plt.legend(ncol=2)

        save_fig(fig, f"power.curve_{save_func(conf,sig)}", save_path)
        plt.close()


def plot_alpha(confs: list[Config], title_func, label_func, save_func, post_func = None, save_path: str = "./power_plots", load_path = "./test_res", line_styles = [], line_colors = []) -> None:

    conf = confs[0]

    makedirs(save_path, exist_ok=True)
    sizes = np.arange(0,conf.sample_size - conf.initial_size + 1)

    def plot_data(sizes, conf, line_style = None, line_color=None):
        name = conf.get_test_name()
        power = np.load(f"{load_path}/power_{name}.npy")
        print(name)
        rcParams["lines.dashed_pattern"] = (5,10)

        if line_style is not None and line_color is not None:
            plt.plot(sizes, power[i], label=label_func(conf), color=line_color, linestyle=line_style)
        elif line_style is not None:
            plt.plot(sizes, power[i], label=label_func(conf), linestyle=line_style)
        elif line_color is not None:
            plt.plot(sizes, power[i], label=label_func(conf), color=line_color)
        else:
            plt.plot(sizes, power[i], label=label_func(conf))
    
    for i, sig in enumerate(confs[0].significance):
        fig, axs = create_fig(hfrac=0.85)

        if line_styles and line_colors:
            for conf, line_style, line_color in zip(confs, line_styles, line_colors):
                plot_data(sizes, conf, line_style, line_color)
        elif line_styles:
            for conf, line_style in zip(confs, line_styles):
                plot_data(sizes, conf, line_style = line_style)
        elif line_colors:
            for conf, line_color in zip(confs, line_colors):
                plot_data(sizes, conf, line_color = line_color)
        else:
            for conf in confs:
                plot_data(sizes, conf)

        plt.plot([0,sizes[-1]], [sig,sig], label=rf"sign. $\alpha={sig}$")
            
        plt.title(title_func(conf, sig))
        plt.xlabel("Iterations")
        plt.ylabel("Type I Error")
        plt.ylim(0, sig + 0.4*sig)
        plt.grid()

        if post_func is not None:
            post_func(fig, axs, confs, sig)
        else:
            plt.legend(ncol=2)

        save_fig(fig, f"alpha.curve_{save_func(conf,sig)}", save_path)
        plt.close()