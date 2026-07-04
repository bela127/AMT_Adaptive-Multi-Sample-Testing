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
    m = 1
    max_iterations = 2000
    common_p = 0.2
    reps = 2500
    p_diff = 0.1
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
        power = np.sum(test_decision, axis=0) / num_reps
        
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
        
        ax.plot(np.arange(0, num_iters), power, label=display_label, markersize=5, **style)
        plotted_any = True

    if plotted_any:
        ax.set_title(f"Empirical Power Curves ($N={n}$, Selection: `{sel_mode}`, $H_1: \\Delta p={p_diff}$)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Sequential Sample Observations ($t$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Statistical Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max_iterations)
        plot_utils.apply_standard_legend(ax)
        plt.savefig(f"{plot_utils.SAVE_PATH}/power_comparison_n{n}_{sel_mode}_{common_p}_{p_diff}.png", bbox_inches='tight')