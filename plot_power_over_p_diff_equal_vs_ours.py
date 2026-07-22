import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs
from amt.configuration import Config
import plot_utils

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    plot_utils.LOAD_PATH = "./exp_results/test_res/vari_p_diff"
    
    # 1. Define configuration settings matching your ablation run
    n = 20
    m = 1
    max_iterations = 2000
    common_p = 0.5
    reps = 2500
    
    p_diff_values = [0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
    
    # 2. Define the specific target configurations you requested:
    # Format: (selection_mode, test_mode)
    target_configs = [
        ("beta.med", "betabinom.pmf"),
        ("equal", "betabinom.pmf"),
        #("beta.med", "lil.variance"),
        #("beta.med", "hoeffding.variance.infinite"),
        ("equal", "chi2"),
        #("beta.med", "bernstein.empiric.infinite"),
        #("beta.med", "kl.horizon"),
        ("equal", "kw"),
        #("beta.med", "betting.e.variable"),
    ]
    
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    plotted_any = False  
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    # 3. Iterate through our target pairs
    for sel_mode, test_mode in target_configs:
        x_points, y_power_points = [], []

        for val in p_diff_values:
            
            dummy_conf = Config(
                n=n, 
                m=m, 
                hyp=1,
                sample_size=max_iterations, 
                initial_size=10, 
                reps=reps,
                common_p=common_p, 
                p_diff=val, 
                selection_mode=sel_mode, 
                test_mode=test_mode,
                coin_weights="posdif"
            )
            
            # Use the designated output folder where your test results were saved
            file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
            
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
                marker_stride=1,
                group_offset = True
            )
            group_counters[group_key] += 1 
            line_index += 1
            
            display_sel_label, group_key = plot_utils.resolve_selection_metadata(sel_mode)
            # Format display label to show both selection policy and test type
            combined_label = f"{display_test_label} + {display_sel_label}"
            
            ax.plot(x_points, y_power_points, label=combined_label, markersize=6, **style)
            plotted_any = True

    if plotted_any:
        ax.set_title(
            f"End Power across Effect Sizes for Clasic Tests\n"
            f"($K={n}, M={m}, p = {common_p}$ at $T={max_iterations}$)", 
            fontsize=13, fontweight='bold', pad=15
        )
        ax.set_xlabel(r"Effect Size ($\Delta p$)", fontsize=11, labelpad=8)
        ax.set_ylabel("End Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(p_diff_values) 
        
        plot_utils.apply_standard_legend(ax, ncol=3)
        
        save_filename = f"power_over_p_diff_for_classic"

        plot_utils.save_fig(fig, save_filename, plot_utils.SAVE_PATH, "svg")
        plot_utils.save_fig(fig, save_filename, plot_utils.SAVE_PATH, "png")
        plt.close()
        print(f"Power plot successfully generated and saved as: {save_filename}")