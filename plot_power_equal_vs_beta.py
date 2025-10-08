from os import makedirs

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams


from amt.configuration import Config
from amt.utils import translate_names, create_fig, save_fig, plot_power, translate_test_colors


def post(fig, ax, confs, sig):
    """
    Creates a complex legend from existing line plots on the given axes.

    The legend is structured as:
    '[line marker 1] Equal vs [line marker 2] Beta Selection for [conf.testmode] test.'
    """
    # Ensure ax is an Axes object, not an array of axes if there's only one subplot.
    # This handles cases where plt.subplots() returns (fig, ax) or (fig, [ax])
    if isinstance(ax, (list, tuple)):
        if len(ax) == 1:
            ax = ax[0]
        else:
            print("Warning: Multiple axes detected. Using the first axes for legend generation.")
            ax = ax[0] # Or handle multiple axes as per your specific needs

    # Fetch existing line plots from the axes
    # ax.get_lines() returns a list of all Line2D objects plotted on the axes.
    all_lines = ax.get_lines()

    # Find the specific lines we want for our custom legend based on their labels
    equal_label = "Equal selection"
    beta_label = "Beta selection"

    equal_lines = []
    beta_lines = []

    for line in all_lines:
        if equal_label in line.get_label():
            equal_lines.append(line)
        elif beta_label in line.get_label():
            beta_lines.append(line)


    # Construct the custom legend entries
    # We need to create custom handles and labels for the legend.
    # The handles and labels for the legend
    legend_handles = []
    legend_labels = []

    for equal_line, conf in zip(equal_lines, confs[1::2]):

        # Extract properties for the 'Equal' line
        marker1 = equal_line.get_marker()
        color1 = equal_line.get_color()
        linestyle1 = equal_line.get_linestyle()

        # Create dummy handles that represent the lines' appearances for the legend.
        # These are used to display the correct marker and color in the legend entry.
        handle1 = mlines.Line2D([], [], color=color1, marker=marker1, linestyle=linestyle1, markersize=8)
        handle1.set_dashes((3.7, 1.6))

        # Get the test mode from the configuration dictionary
        test_name, sel_name = translate_names(conf.test_mode, conf.selection_mode)

        # Construct the custom label string
        custom_label1 = f"Equal vs "

        # Assign the handles and labels for the legend
        legend_handles.append(handle1)

        legend_labels.append(custom_label1)

    for beta_selection_line, conf in zip(beta_lines, confs[1::2]):

        # Extract properties for the 'Beta Selection' line
        marker2 = beta_selection_line.get_marker()
        color2 = beta_selection_line.get_color()
        linestyle2 = beta_selection_line.get_linestyle()

        # Create dummy handles that represent the lines' appearances for the legend.
        # These are used to display the correct marker and color in the legend entry.
        handle2 = mlines.Line2D([], [], color=color2, marker=marker2, linestyle=linestyle2, markersize=8)

        # Get the test mode from the configuration dictionary
        test_name, sel_name = translate_names(conf.test_mode, conf.selection_mode)

        # Construct the custom label string
        custom_label2 = f"Beta Selection for {test_name} test."

        # Assign the handles and labels for the legend
        legend_handles.append(handle2)

        legend_labels.append(custom_label2)

    # Add the legend to the plot
    # Using `bbox_to_anchor` and `loc` to position the legend outside the plot area
    ax.legend(handles=legend_handles, labels=legend_labels, ncol=2, columnspacing=0.5)

if __name__ == "__main__":       
    for n in [5, 10, 15, 20]:#5, 10, 15, 20
        confs = []
        line_styles = []
        line_colors = []
        for test_mode in ["mean", "chi2","kw", "beta", "betabinom.comb"]:#"mean", "chi2","kw", "beta", "betabinom.comb"
            for sel_mode in ["equal", "beta.med"]:# 
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
                
                if sel_mode == "equal":
                    line_style = 'dashed'
                else:
                    line_style = 'solid'
                line_color = translate_test_colors(test_mode)

                line_styles.append(line_style)
                line_colors.append(line_color)

            def title(conf: Config, sig):
                return rf"Equal vs Beta selection at $\alpha={sig}$ for {conf.n} coins "
            def label(conf: Config):
                test_name, sel_name = translate_names(conf.test_mode, conf.selection_mode)
                return f"{sel_name} selection, {test_name} test"
            def save(conf: Config, sig):
                return f"test.mode-equal.vs.beta_coins-{conf.n}_sig-{sig}"
            
            plot_power(confs, title_func=title, label_func=label, save_func=save, post_func=post, load_path="./test_res", save_path = "./power_plots/equal.vs.beta", line_colors=line_colors, line_styles=line_styles)
