import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs
from amt.configuration import Config
import plot_utils

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    # Add other test mode strings here to compare them side-by-side
    test_modes = ["betabinom.pmf"] 
    sel_mode = "beta.med"
    m = 1
    max_iterations = 2000
    default_common_p = 0.5
    default_p_diff = 0.05
    reps = 2500
    
    n_values = [5, 10, 15, 20, 30, 40]
    load_path_dir = "./exp_results/test_res/vari_n"
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False  
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    manual_label_overrides = {
        "betabinom.pmf": "BetaBinom PMF",
        "betabinom.comb": "BetaBinom Combined",
        "kw": "Kruskal-Wallis",
        "chi2": "Chi-Squared",
        "mean": "Empirical Mean",
        "beta": "Beta Mixture"
    }

    for test_mode in test_modes:
        x_points, y_power_points = [], []
        for n in n_values:
            dummy_conf = Config(
                n=n, m=m, hyp=m, sample_size=max_iterations, initial_size=10, 
                reps=reps, common_p=default_common_p, p_diff=default_p_diff, 
                selection_mode=sel_mode, test_mode=test_mode
            )
            file_path = f"{load_path_dir}/reject_{dummy_conf.get_test_name()}.npy"
            
            # Straightforward matrix parsing without index slicing
            processed = plot_utils.load_and_process_results(file_path, test_mode)
            if processed is not None:
                test_decision, num_reps, _ = processed
                x_points.append(n)
                y_power_points.append(np.sum(test_decision[:, -1], axis=0) / num_reps)

        if x_points:
            display_label, group_key = plot_utils.resolve_test_metadata(test_mode)
            if "unmapped" in display_label.lower() or "unmappt" in display_label.lower():
                display_label = manual_label_overrides.get(test_mode, test_mode)
                
            style = plot_utils.get_line_style(
                group_key=group_key, group_counter=group_counters.get(group_key, 0), 
                line_index=line_index, color_lookup_table=plot_utils.TEST_PALETTE, marker_stride=1
            )
            if group_key in group_counters: group_counters[group_key] += 1
            line_index += 1
            
            ax.plot(x_points, y_power_points, label=display_label, markersize=6, **style)
            plotted_any = True

    if plotted_any:
        ax.set_title(f"Final Empirical Power vs. Number of Total Coins $N$\n(Selection: `{sel_mode}`, $M={m}$, $\\Delta p={default_p_diff}$ at $t={max_iterations}$)", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("Number of Total Coins ($N$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Statistical Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(n_values) 
        ax.set_xlim(min(n_values) - 1, max(n_values) + 1)
        plot_utils.apply_standard_legend(ax)
        plt.savefig(f"{plot_utils.SAVE_PATH}/final_power_by_n_{sel_mode}_comparison.png", bbox_inches='tight')
        plt.close()