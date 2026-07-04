import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs
from amt.tests import Test
from amt.configuration import Config
import plot_utils

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    test_suite = Test()
    test_modes = list(test_suite.test_modes.keys())

    n, sel_mode, m, max_iterations, default_common_p = 20, "beta.med", 1, 2000, 0.5
    reps_values = [2500, 5000, 2500, 2500, 2500, 2500, 2500]
    p_diff_values = [0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
    
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False  
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    for test_mode in test_modes:
        x_points, y_power_points = [], []

        for val, reps in zip(p_diff_values, reps_values):
            current_m = 0 if val == 0.0 else m
            dummy_conf = Config(n=n, m=current_m, sample_size=max_iterations, initial_size=10, reps=reps,
                                common_p=default_common_p, p_diff=val, hyp=current_m, selection_mode=sel_mode, test_mode=test_mode)
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
        ax.set_title(f"Final Empirical Power ($N={n}$, Selection: `{sel_mode}`, $p_{{common}}={default_common_p}$ at $t={max_iterations}$)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(r"Effect Size Variation ($\Delta p$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Statistical Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(p_diff_values) 
        plot_utils.apply_standard_legend(ax)
        plt.savefig(f"{plot_utils.SAVE_PATH}/final_power_by_p_diff_n{n}_{sel_mode}.png", bbox_inches='tight')