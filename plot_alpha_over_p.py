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

    # Environment Parameters for Global Null (H0)
    n = 20
    sel_mode = "beta.med"
    m = 0
    max_iterations = 2000
    default_p_diff = 0.0
    target_alpha = 0.05  # The nominal Type-I error rate bound
    
    reps_values = [2500, 2500, 2500, 2500, 2500, 2500, 2500] 
    common_p_values = [0.05, 0.2, 0.35, 0.5, 0.6, 0.7, 0.85] 
    
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False  
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    for test_mode in test_modes:
        x_points, y_error_points = [], []

        for val, reps in zip(common_p_values, reps_values):
            dummy_conf = Config(n=n, m=m, sample_size=max_iterations, initial_size=10, reps=reps,
                                common_p=val, p_diff=default_p_diff, hyp=m, selection_mode=sel_mode, test_mode=test_mode)
            file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
            
            processed = plot_utils.load_and_process_results(file_path, test_mode)
            if processed is not None:
                test_decision, num_reps, _ = processed
                
                # Compute empirical Type-I error rate at the terminal observation index step
                final_type1_error = np.sum(test_decision, axis=0) / num_reps
                x_points.append(val)
                y_error_points.append(np.max(final_type1_error[50:-50])) #mean alpha error across all iterations for this test mode and p value

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
            ax.plot(x_points, y_error_points, label=display_label, markersize=6, **style)
            plotted_any = True

    if plotted_any:
        # Draw a stark, highly visible target indicator line at the target nominal alpha threshold
        ax.axhline(y=target_alpha, color='#b71c1c', linestyle='-', linewidth=2.0, alpha=0.8, zorder=10, 
                   label=f"Nominal Alpha Threshold ($\\alpha = {target_alpha}$)")

        # Layout Adjustments for clear canvas styling
        ax.set_title(f"Final Empirical Type-I Error Rates Under Global Null ($N={n}$, Selection: `{sel_mode}`, $\\Delta p={default_p_diff}$ at $t={max_iterations}$)", 
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(r"Base Probability ($p_{{common}}$)", fontsize=11, labelpad=8)
        ax.set_ylabel("False Positive Rate (Empirical $\\alpha$)", fontsize=11, labelpad=8)
        
        # Scale limit vertically above alpha to keep cross-method variability readable
        ax.set_ylim(-0.005, target_alpha * 2.5) 
        ax.set_xticks(common_p_values) 
        
        plot_utils.apply_standard_legend(ax)
        
        save_filename = f"{plot_utils.SAVE_PATH}/final_alpha_by_common_p_n{n}_{sel_mode}.png"
        plt.savefig(save_filename, bbox_inches='tight')
        print(f"Successfully generated Type-I error comparison: {save_filename}")
    
    plt.close(fig)