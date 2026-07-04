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

    n = 20
    sel_mode = "beta.med"
    m = 0
    max_iterations = 2000
    reps = 2500
    common_p = 0.35 #[0.05, 0.2, 0.35, 0.5, 0.6, 0.7, 0.85]
    p_diff = 0
    target_alpha = 0.05
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False  
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    for test_mode in test_modes:
        dummy_conf = Config(n=n, m=m, sample_size=max_iterations, initial_size=10, reps=reps,
                            common_p=common_p, p_diff=p_diff, hyp=m, selection_mode=sel_mode, test_mode=test_mode)
        file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
        
        processed = plot_utils.load_and_process_results(file_path, test_mode)
        if processed is None:
            continue
            
        test_decision, num_reps, num_iters = processed
        type_1_error = np.sum(test_decision, axis=0) / num_reps
        
        # Resolve styling metadata to ensure visual consistency across plots
        display_label, group_key = plot_utils.resolve_test_metadata(test_mode)
        style = style = plot_utils.get_line_style(
        group_key=group_key, 
        group_counter=group_counters[group_key], 
        line_index=line_index, 
        color_lookup_table=plot_utils.TEST_PALETTE,
        )
        group_counters[group_key] += 1 
        line_index += 1
        
        ax.plot(np.arange(0, num_iters), type_1_error, label=display_label, markersize=5, **style)
        plotted_any = True

    if plotted_any:
        ax.axhline(y=target_alpha, color='#b71c1c', linestyle='-', linewidth=2.0, alpha=0.8, zorder=10, 
                   label=f"Nominal Alpha Threshold ($\\alpha = {target_alpha}$)")
        ax.set_title(f"Empirical Type-I Error Rates Under Global Null ($N={n}$, Selection: `{sel_mode}`, $H_0: p_i = {common_p}$)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Sequential Sample Observations ($t$)", fontsize=11, labelpad=8)
        ax.set_ylabel("False Positive Rate (Empirical $\\alpha$)", fontsize=11, labelpad=8)
        ax.set_ylim(-0.005, target_alpha * 2.5) 
        ax.set_xlim(0, max_iterations)
        plot_utils.apply_standard_legend(ax)
        plt.savefig(f"{plot_utils.SAVE_PATH}/alpha_comparison_n{n}_{sel_mode}_{common_p}_{p_diff}.png", bbox_inches='tight')