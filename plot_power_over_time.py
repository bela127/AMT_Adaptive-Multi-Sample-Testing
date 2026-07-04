import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs
from amt.tests import Test
from amt.configuration import Config
import plot_utils

if __name__ == "__main__":
    # Set the standard theme
    sns.set_theme(style="whitegrid")
    
    test_suite = Test()
    test_modes = list(test_suite.test_modes.keys())
    
    # Configuration parameters
    time_load_path = "./time_benchmark"
    n = 20
    sel_mode, m, max_iterations, default_common_p = "beta.med", 1, 2000, 0.5
    target_p_diff = 0.1
    target_reps = 2500
    
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    # Initialize a 1x2 subplot figure to hold both comparisons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
    plotted_any = False
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    for test_mode in test_modes:
        # 1. Extract execution time and compute the average across all N configurations
        time_file_path = f"{time_load_path}/bench_{test_mode}.npy"
        try:
            runtimes = np.load(time_file_path)
            avg_runtime = np.mean(runtimes)
        except FileNotFoundError:
            print(f"WARNING: Missing time file {time_file_path}. Skipping {test_mode}.")
            continue
            
        # 2. Extract statistical power for p_diff = 0.1
        current_m = 0 if target_p_diff == 0.0 else m
        dummy_conf = Config(n=n, m=current_m, sample_size=max_iterations, initial_size=10, reps=target_reps,
                            common_p=default_common_p, p_diff=target_p_diff, hyp=current_m, selection_mode=sel_mode, test_mode=test_mode)
        power_file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
        
        processed = plot_utils.load_and_process_results(power_file_path, test_mode)
        if processed is not None:
            test_decision, num_reps, _ = processed
            power = np.sum(test_decision[:, -1], axis=0) / num_reps
            
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
            
            # Safe clean-up: Remove both variants if present, then set to 'None'
            style.pop('linestyle', None)
            style.pop('ls', None)
            style['linestyle'] = 'None'
            
            # --- Calculation Scaling Logic ---
            # Checking against plot_utils.FIXED_BATCH_MODES since it's defined in utils
            if test_mode in plot_utils.FIXED_BATCH_MODES:
                # Computed once at the end of the batch horizon
                scaled_runtime = avg_runtime 
            else:
                # Anytime tests are evaluated at each sequential iteration step
                scaled_runtime = avg_runtime * max_iterations
            
            # Subplot 1: Raw Base Benchmark Time Cost
            ax1.plot(avg_runtime, power, label=display_label, markersize=10, **style)
            
            # Subplot 2: Scaled Time Cost (Reflecting Sequential Evaluation Load)
            ax2.plot(scaled_runtime, power, label=display_label, markersize=10, **style)
            
            plotted_any = True

    # Finalize and polish both subplots
    if plotted_any:
        # Title adjustments
        ax1.set_title("Empirical Power vs. Base Execution Time", fontsize=13, fontweight='bold', pad=12)
        ax2.set_title(f"Empirical Power vs. Computation Scaled Time ($T={max_iterations}$)", fontsize=13, fontweight='bold', pad=12)
        
        # Consistent labels & scaling configurations
        for ax in [ax1, ax2]:
            ax.set_ylabel("Statistical Power ($1 - \\beta$)", fontsize=11, labelpad=8)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            
        ax1.set_xlabel("Average Execution Time (Seconds per 1,000 evaluations)", fontsize=11, labelpad=8)
        ax2.set_xlabel("Total Experiment Execution Time Cost (Seconds)", fontsize=11, labelpad=8)
        
        # Center the line-free legend beneath the side-by-side layout panel
        plot_utils.apply_scatter_legend(ax1, bbox_to_anchor=(1.05, -0.15))
        
        # Overall figure meta-title
        fig.suptitle(f"Algorithmic Efficiency Tradeoffs ($N={n}$, $\\Delta p={target_p_diff}$, Selection: `{sel_mode}`)", 
                     fontsize=15, fontweight='bold', y=1.02)
        
        plt.savefig(f"{plot_utils.SAVE_PATH}/power_vs_time_dual_aspect_n{n}.png", bbox_inches='tight')