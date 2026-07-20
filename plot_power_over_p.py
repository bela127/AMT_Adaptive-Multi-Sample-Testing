import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs
from amt.tests import Test
from amt.configuration import Config
import plot_utils

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    plot_utils.LOAD_PATH = "./exp_results/test_res/p_and_p_diff"
    
    test_suite = Test()
    test_modes = list(test_suite.test_modes.keys())

    n = 20
    sel_mode = "beta.med"
    m = 1
    max_iterations = 2000
    default_p_diff = 0.1
    reps_values = [2500, 2500, 2500, 2500, 2500, 2500, 2500] 
    common_p_values = [0.05, 0.2, 0.35, 0.5, 0.6, 0.7, 0.85] 
    
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False  
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    for test_mode in test_modes:
        x_points, y_power_points = [], []

        for val, reps in zip(common_p_values, reps_values):
            current_m = 0 if default_p_diff == 0.0 else m
            dummy_conf = Config(n=n, m=current_m, sample_size=max_iterations, initial_size=10, reps=reps,
                                common_p=val, p_diff=default_p_diff, hyp=current_m, selection_mode=sel_mode, test_mode=test_mode)
            file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
            
            processed = plot_utils.load_and_process_results(file_path, test_mode)
            if processed is not None:
                test_decision, num_reps, _ = processed
                x_points.append(val)
                y_power_points.append(np.sum(test_decision[:, -1], axis=0) / num_reps)

        if x_points:
            # Resolve styling metadata to ensure visual consistency across plots
            display_label, group_key = plot_utils.resolve_test_metadata(test_mode)
            style = style = plot_utils.get_line_style(
            group_key=group_key, 
            group_counter=group_counters[group_key], 
            line_index=line_index, 
            color_lookup_table=plot_utils.TEST_PALETTE,
            marker_stride=1
            )
            group_counters[group_key] += 1 
            line_index += 1
            ax.plot(x_points, y_power_points, label=display_label, markersize=6, **style)
            plotted_any = True

    if plotted_any:
        ax.set_title(f"Final Empirical Power ($N={n}$, Selection: `{sel_mode}`, $\\Delta p={default_p_diff}$ at $t={max_iterations}$)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(r"Base Probability ($p_{{common}}$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Statistical Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(common_p_values) 
        plot_utils.apply_standard_legend(ax)
        plt.savefig(f"{plot_utils.SAVE_PATH}/final_power_by_common_p_n{n}_{sel_mode}.png", bbox_inches='tight')