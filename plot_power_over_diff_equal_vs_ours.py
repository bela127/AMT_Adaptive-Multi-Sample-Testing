import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs
from amt.configuration import Config
import plot_utils

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    # 1. Define configuration settings matching your ablation run
    n = 20
    max_iterations = 2000
    default_common_p = 0.5
    reps = 2500
    
    p_diff_values = [0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
    
    # 2. Define the specific target configurations you requested:
    # Format: (selection_mode, test_mode)
    target_configs = [
        ("beta.med", "betabinom.pmf"),
        ("equal", "betabinom.pmf"),
        ("equal", "chi2"),
        ("equal", "kw")
    ]
    
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False  
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    # 3. Iterate through our target pairs
    for sel_mode, test_mode in target_configs:
        x_points, y_power_points = [], []

        for val in p_diff_values:
            
            dummy_conf = Config(
                n=n, 
                m=1, 
                hyp=1,
                sample_size=max_iterations, 
                initial_size=10, 
                reps=reps,
                common_p=default_common_p, 
                p_diff=val, 
                selection_mode=sel_mode, 
                test_mode=test_mode,
                coin_weights="posdif"
            )
            
            # Use the designated output folder where your test results were saved
            file_path = f"./exp_results/test_res/reject_{dummy_conf.get_test_name()}.npy"
            
            processed = plot_utils.load_and_process_results(file_path, test_mode)
            if processed is not None:
                test_decision, num_reps, _ = processed
                x_points.append(val)
                # Capture final power at terminal index t = -1
                y_power_points.append(np.sum(test_decision[:, -1], axis=0) / num_reps)

        if x_points:
            # Resolve styling metadata for consistency using your helper methods
            display_test_label, group_key = plot_utils.resolve_test_metadata(test_mode)
            style = plot_utils.get_line_style(
                group_key=group_key, 
                group_counter=group_counters[group_key], 
                line_index=line_index, 
                color_lookup_table=plot_utils.TEST_PALETTE,
                marker_stride=1
            )
            group_counters[group_key] += 1 
            line_index += 1
            
            # Format display label to show both selection policy and test type
            combined_label = f"{display_test_label} ({sel_mode})"
            
            ax.plot(x_points, y_power_points, label=combined_label, markersize=6, **style)
            plotted_any = True

    if plotted_any:
        ax.set_title(
            f"Ablation Study: Terminal Power Across Effect Sizes\n"
            f"($N={n}$, $p_{{common}}={default_common_p}$ at $t={max_iterations}$)", 
            fontsize=13, fontweight='bold', pad=15
        )
        ax.set_xlabel(r"Effect Size Variation ($\Delta p$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Statistical Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(p_diff_values) 
        
        plot_utils.apply_standard_legend(ax)
        
        save_filename = f"ablation_power_by_p_diff_n{n}_mixed_selection.png"
        plt.savefig(f"{plot_utils.SAVE_PATH}/{save_filename}", bbox_inches='tight')
        plt.close()
        print(f"Power plot successfully generated and saved as: {save_filename}")