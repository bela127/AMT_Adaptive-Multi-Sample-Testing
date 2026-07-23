import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs

from amt.tests import Test
from amt.configuration import Config
import plot_utils

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    test_modes = list(plot_utils.TEST_GROUPS.keys())
    
    # Configuration parameters
    plot_utils.LOAD_PATH = "./exp_results/test_res/p_and_p_diff"
    time_load_path = "./exp_results/time_benchmark"
    
    n, sel_mode, m, max_iterations, default_common_p = 20, "beta.med", 1, 2000, 0.5
    target_p_diff = 0.1
    target_reps = 2500
    
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    # Initialize a 1x2 subplot figure to hold both comparisons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=300)
    plotted_any = False
    
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    for test_mode in test_modes:
        # 1. Extract execution time and compute the average across all configurations
        time_file_path = f"{time_load_path}/bench_{test_mode}.npy"
        try:
            runtimes = np.load(time_file_path)
            avg_runtime = np.mean(runtimes)
        except FileNotFoundError:
            print(f"WARNING: Missing time file {time_file_path}. Skipping {test_mode}.")
            continue
            
        # 2. Extract statistical power for p_diff = 0.1
        current_m = 0 if target_p_diff == 0.0 else m
        dummy_conf = Config(
            n=n, m=current_m, sample_size=max_iterations, initial_size=10, reps=target_reps,
            common_p=default_common_p, p_diff=target_p_diff, hyp=current_m, 
            selection_mode=sel_mode, test_mode=test_mode
        )
        power_file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
        
        processed = plot_utils.load_and_process_results(power_file_path, test_mode)
        if processed is not None:
            test_decision, num_reps, _ = processed
            power = np.sum(test_decision[:, -1], axis=0) / num_reps
            
            # Resolve styling metadata to ensure visual consistency across plots
            display_label, group_key = plot_utils.resolve_test_metadata(test_mode)
            style = plot_utils.get_line_style(
                group_key=group_key, 
                group_counter=group_counters[group_key], 
                line_index=line_index, 
                color_lookup_table=plot_utils.TEST_PALETTE,
                marker_stride=1
            )
            group_counters[group_key] += 1 
            line_index += 1
            
            # Clean-up linestyle for scatter points
            style.pop('linestyle', None)
            style.pop('ls', None)
            style['linestyle'] = 'None'
            
            # --- Calculation Scaling Logic ---
            if test_mode in plot_utils.FIXED_BATCH_MODES:
                scaled_runtime = avg_runtime 
            else:
                scaled_runtime = avg_runtime * max_iterations
            
            # Subplot 1: Raw Base Benchmark Time Cost
            ax1.plot(avg_runtime, power, label=display_label, markersize=8, **style)
            
            # Subplot 2: Scaled Time Cost (Reflecting Sequential Evaluation Load)
            ax2.plot(scaled_runtime, power, label=display_label, markersize=8, **style)


            # HACK: Nach "Unserem Ansatz" einen Spaltenumbruch in der Legende erzwingen
            # (Fügt leere Platzhalter ein, die genau 1 Zeile über alle `ncol` Spalten auffüllen)
            if test_mode == "betabinom.pmf":  # Oder die entsprechende Key-Bedingung für deinen Ansatz
                for _ in range(3):  # Da ncol=3 ist
                    ax1.plot([], [], label=" ", color="none")
            
            plotted_any = True

    # Finalize and polish both subplots
    if plotted_any:
        selection_display_name = plot_utils.resolve_selection_metadata(sel_mode)[0]
        
        # Subplot Titles
        ax1.set_title("Power vs. Base Execution Time", fontsize=12, fontweight='bold', pad=12)
        ax2.set_title(f"Power vs. Scaled Sequential Time ($T={max_iterations}$)", fontsize=12, fontweight='bold', pad=12)
        
        # Consistent labels & scaling configurations
        for ax in [ax1, ax2]:
            ax.set_ylabel("End Power ($1 - \\beta$)", fontsize=11, labelpad=8)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            
        ax1.set_xlabel("Average Execution Time (s / 1k evaluations)", fontsize=11, labelpad=8)
        ax2.set_xlabel("Total Computation Time Cost (s)", fontsize=11, labelpad=8)
        
        # Zentrierte Platzierung der Scatter-Legende unter den beiden Subplots
        plot_utils.apply_scatter_legend(
            ax1, 
            bbox_to_anchor=(1.1, -0.18), 
            loc="upper center", 
            ncol=5,
            )
        
        # Overall figure meta-title (N -> K Konsistenz)
        fig.suptitle(
            f"Algorithmic Efficiency Tradeoffs for Selection {selection_display_name}\n"
            f"($K={n}$, $M={m}$, $p={default_common_p}$, $\\Delta p={target_p_diff}$)", 
            fontsize=13, fontweight='bold', y=1.06
        )
        
        save_name = f"power_vs_time_dual_aspect_n{n}"
        plot_utils.save_fig(fig, save_name, plot_utils.SAVE_PATH, "svg")
        plot_utils.save_fig(fig, save_name, plot_utils.SAVE_PATH, "png")
        plt.close()
        print(f"Erfolgreich gespeichert unter: {save_name}")