import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from os import makedirs

from amt.configuration import Config
import plot_utils

OUR_APPROACH_KEY = "betabinom.pmf"
TARGET_POWER_LEVEL = 0.60
INITIAL_SIZE_PER_ARM = 10
SAMPLES_PER_ITER = 2

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    plot_utils.LOAD_PATH = "./exp_results/test_res/p_and_p_diff"
    test_modes = list(plot_utils.TEST_GROUPS.keys())

    n, sel_mode, m, max_iterations, default_common_p = 20, "beta.med", 1, 2000, 0.5
    p_diff_values = [0.1, 0.125, 0.15, 0.2]
    
    init_sample_offset = n * INITIAL_SIZE_PER_ARM  # 20 * 10 = 200 Samples
    max_total_samples = init_sample_offset + (max_iterations * SAMPLES_PER_ITER)
    
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    # ---------------------------------------------------------------
    # 1. Pre-pass: Store power curves for our approach
    # ---------------------------------------------------------------
    our_power_curves = {}
    
    for val in p_diff_values:
        current_m = 0 if val == 0.0 else m
        dummy_conf = Config(
            n=n, m=current_m, sample_size=max_iterations, initial_size=INITIAL_SIZE_PER_ARM,
            common_p=default_common_p, p_diff=val, hyp=current_m, 
            selection_mode=sel_mode, test_mode=OUR_APPROACH_KEY
        )
        file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
        processed = plot_utils.load_and_process_results(file_path, OUR_APPROACH_KEY)
        
        if processed is not None:
            test_decision, num_reps, _ = processed
            our_power_curves[val] = np.sum(test_decision, axis=0) / num_reps

# ---------------------------------------------------------------
    # 2. Plot Our Approach FIRST (Baseline reference line at y = 1.0)
    # ---------------------------------------------------------------
    our_display_label, _ = plot_utils.resolve_test_metadata(OUR_APPROACH_KEY)
    valid_x = [val for val in p_diff_values if val in our_power_curves]

    ax.axhline(
        1.0, 
        color='gray', 
        linestyle='--', 
        linewidth=1.5, 
        alpha=0.7, 
        zorder=5, 
        label=f"{our_display_label} (Reference)"
    )
    # ---------------------------------------------------------------
    # 3. Main Loop: Compute ratio for all other approaches
    # ---------------------------------------------------------------
    for test_mode in test_modes:
        if test_mode == OUR_APPROACH_KEY:
            continue

        x_points, y_ratio_points = [], []

        for val in p_diff_values:
            if val not in our_power_curves:
                continue
                
            current_m = 0 if val == 0.0 else m
            dummy_conf = Config(
                n=n, m=current_m, sample_size=max_iterations, initial_size=INITIAL_SIZE_PER_ARM,
                common_p=default_common_p, p_diff=val, hyp=current_m, 
                selection_mode=sel_mode, test_mode=test_mode
            )
            file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
            
            processed = plot_utils.load_and_process_results(file_path, test_mode)
            if processed is None:
                continue

            test_decision, num_reps, num_iters = processed
            base_power_over_time = np.sum(test_decision, axis=0) / num_reps
            our_power_over_time = our_power_curves[val]

            reached_base = np.where(base_power_over_time >= TARGET_POWER_LEVEL)[0]
            reached_ours = np.where(our_power_over_time >= TARGET_POWER_LEVEL)[0]

            # Keep x-axis continuous
            x_points.append(val)

            # Insert ratio if BOTH reached target power; otherwise insert NaN to break the line
            if len(reached_ours) > 0 and len(reached_base) > 0:
                samples_ours = init_sample_offset + (reached_ours[0] * SAMPLES_PER_ITER)
                samples_base = init_sample_offset + (reached_base[0] * SAMPLES_PER_ITER)
                y_ratio_points.append(samples_base / samples_ours)
            else:
                y_ratio_points.append(np.nan)

        # Plot only if there is at least one valid numerical evaluation
        if len(y_ratio_points) > 0 and not np.all(np.isnan(y_ratio_points)):
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
            
            # Matplotlib will render gaps automatically over np.nan entries
            ax.plot(x_points, y_ratio_points, label=display_label, markersize=6, **style)
# ---------------------------------------------------------------
    # 4. Axes & Logarithmic Formatting
    # ---------------------------------------------------------------
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    ax.set_yscale("log")

    # Explicitly set y-limits covering your range (~1 to 6)
    ax.set_ylim(0.95, 6.0)

    # Place major ticks at whole numbers and minor ticks at 0.5 increments
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5]))
    ax.yaxis.set_minor_locator(FixedLocator([1.5, 2.5, 3.5, 4.5, 5.5]))

    # Turn off Matplotlib's default log log-tick formatters (which introduce 10^0 / scientific notation)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())

    # Prevent scientific notation offset (e.g., +1e0)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_minor_formatter().set_scientific(False)

    # Enable both major and minor grids for log visibility
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.5)
    
    selection_display_name = plot_utils.resolve_selection_metadata(sel_mode)[0]
    
    ax.set_title(
        f"Relative Data Overhead Across Effect Sizes for Target Power: {int(TARGET_POWER_LEVEL*100)}\\%\n"
        f"($K={n}$, $M={m}$, $p={default_common_p}$ with Selection {selection_display_name} at $T=2000$)", 
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xlabel(r"Effect Size ($\Delta p$)", fontsize=11, labelpad=8)
    ax.set_ylabel(r"Relative Data Overhead ($\frac{\text{Competitor Samples}}{\text{Reference Samples}}$)", fontsize=11, labelpad=8)
    
    ax.set_xticks(p_diff_values)
    plot_utils.apply_standard_legend(ax)
    
    plot_name = "data_loss_ratio_over_p_diff"
    plot_utils.save_fig(fig, plot_name, plot_utils.SAVE_PATH, "svg")
    plot_utils.save_fig(fig, plot_name, plot_utils.SAVE_PATH, "png")
    plt.close()
    print(f"Successfully saved plot as: {plot_name}")