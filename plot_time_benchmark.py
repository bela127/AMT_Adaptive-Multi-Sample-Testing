import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs
from amt.tests import Test
import plot_utils

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    test_suite = Test()
    test_modes = list(test_suite.test_modes.keys())
    
    load_path, save_path = "./time_benchmark", "./bandit_res"
    makedirs(save_path, exist_ok=True)
    
    try:
        n_values = np.load(f"{load_path}/benchmark_n_values.npy")
    except FileNotFoundError as e:
        print(f"Error loading configuration baseline metadata: {e}", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    for test_mode in test_modes:
        file_path = f"{load_path}/bench_{test_mode}.npy"
        try:
            runtimes = np.load(file_path)
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
            
            ax.plot(n_values, runtimes, label=display_label, markersize=6, **style)
            plotted_any = True
        except FileNotFoundError:
            print(f"WARNING: Missing file {file_path}. Skipping this data approach loop.")
            continue

    if plotted_any:
        ax.set_title("Algorithmic Test Complexity Scalability (Sequential State Tracking)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Number of Coins ($n$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Normalized Execution Time (Seconds per 1,000 evaluations)", fontsize=11, labelpad=8)
        ax.set_xticks(n_values)
        ax.set_yscale("log")
        plot_utils.apply_standard_legend(ax, bbox_to_anchor=(0.5, -0.12))
        plt.savefig(f"{save_path}/stateful_test_complexity_benchmark.png", bbox_inches='tight')