import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs

# Dynamic imports from your target structures
from amt.bandit_bounds import bandit_bounds
from amt.configuration import Config
import plot_utils
from plot_utils import TEST_GROUPS, apply_scatter_legend

# --- SIMULATION CONFIG PROPERTIES ---
n, m = 20, 1
max_iterations = 2000
common_p, p_diff = 0.5, 0.1
reps = 2500

# 1. Establish the evaluation baseline test mode
target_test_modes = [
    "hoeffding.variance.infinite",
    "betting.e.variable",
    "beta.mixture.infinite",
    "lil.variance",
    "bayesian.beta",
    "glrt.horizon", 
    "one.vs.rest.beta.mixture", 
    "betabinom.pmf"
]

# 2. Extract active key arrays straight from the bandit_bounds repository
active_bandit_bounds = list(bandit_bounds.keys())

# 3. Define the core selection primitives to evaluate
base_non_bandit_modes = ["beta.med", "beta.max", "evar"]
bandit_allocation_mechanisms = ["bandit.max"]#, "bandit.ratio"] # "bandit.max" , "bandit.ratio"


def build_and_plot_merged_end_power(evaluation_queue):
    # Retrieve clean categorical display names for the requested test modes
    ordered_display_names = [TEST_GROUPS[mode][0] for mode in target_test_modes if mode in TEST_GROUPS]
    
    # Store coordinates and labels dynamically as data points successfully resolve
    plotted_points = []  # Will contain dict elements: {'test_idx': int, 'label': str, 'power': float, 'style': dict}
    unique_labels = {}   # Dedup dictionary to store a template style configuration for each display label

    # Loop through all possible combinations exactly as your data loop does
    for t_idx, target_test_mode in enumerate(target_test_modes):
        
        # Fresh counters per test group to align color cycles naturally
        group_counters = {g_key: 0 for g_key in plot_utils.SELECTION_PALETTE.keys()}
        line_index = 0

        for sel_mode, b_kind in evaluation_queue:
            # Build the exact signature configuration
            dummy_conf = Config(
                n=n, m=m, sample_size=max_iterations, initial_size=10, reps=reps,
                common_p=common_p, p_diff=p_diff, hyp=m, 
                selection_mode=sel_mode, test_mode=target_test_mode
            )
            
            # Override the bound identifier context if evaluating a bandit block
            if b_kind is not None:
                dummy_conf.bandit_kind = b_kind
                plot_utils.LOAD_PATH = "./exp_results/test_res/bandit_sel"
            else:
                plot_utils.LOAD_PATH = "./exp_results/test_res/non_bandit_selection"
                
            file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
            
            # Load profile matrix data cleanly using the original helper function
            processed = plot_utils.load_and_process_results(file_path, target_test_mode)
            if processed is None:
                print(f"Warning: No results found for {dummy_conf.get_test_name()}, skipping this configuration.")
                continue
                
            test_decision, num_reps, num_iters = processed
            
            # Calculate final terminal end power vector index instead of the full line
            end_power = np.sum(test_decision[:, -1]) / num_reps
            
            # Resolve labels and colors using selection-specific lookups
            display_label, group_key = plot_utils.resolve_selection_metadata(sel_mode, b_kind)
                
            style = plot_utils.get_line_style(
                group_key=group_key, 
                group_counter=group_counters[group_key], 
                line_index=line_index, 
                color_lookup_table=plot_utils.SELECTION_PALETTE
            )
            
            group_counters[group_key] += 1 
            line_index += 1
            
            # Cache style metadata template for rendering down below
            if display_label not in unique_labels:
                unique_labels[display_label] = style
            
            plotted_points.append({
                'test_idx': t_idx,
                'label': display_label,
                'power': end_power
            })

    if len(plotted_points) == 0:
        print("No valid results loaded to generate the scatter visualization.")
        return

    # Initialize a clean, wide canvas layout 
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Track the exact unique sorted labels seen in the dataset to calculate spacing lanes
    distinct_labels = list(unique_labels.keys())
    num_tests = len(target_test_modes)
    num_labels = len(distinct_labels)
    
    # Configure the geometric layout parameters to fan out (dodge) entries per test category
    x_base_indices = np.arange(num_tests)
    total_lane_width = 0.5
    offsets = np.linspace(-total_lane_width / 2, total_lane_width / 2, num_labels)
    label_to_offset_idx = {lbl: idx for idx, lbl in enumerate(distinct_labels)}

    # Group plot outputs by label explicitly to build the legend cleanly without duplication
    for lbl in distinct_labels:
        # Filter all matching coordinates for this selection strategy
        matching_pts = [p for p in plotted_points if p['label'] == lbl]
        if not matching_pts:
            continue
            
        x_coords = []
        y_coords = []
        
        for pt in matching_pts:
            # Map index to base position + its respective offset lane slice
            offset_val = offsets[label_to_offset_idx[lbl]]
            fanned_x = pt['test_idx'] + offset_val
            
            x_coords.append(fanned_x)
            y_coords.append(pt['power'])
            
        # Extract native colors and assigned markers from original styles dictionary cleanly
        orig_style = unique_labels[lbl]
        marker_color = orig_style.get('color', 'black')
        marker_shape = orig_style.get('marker', 'o')
        
        ax.scatter(
            x_coords, y_coords, 
            label=lbl, 
            color=marker_color, 
            marker=marker_shape,
            s=65, 
            alpha=0.85, 
            edgecolors='none',
            zorder=3
        )

    # Labels and Structural styling adjustments
    ax.set_title(
        f"Terminal Selection Power Comparison Across Anytime Tests ($N={n}$)", 
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_ylabel(r"End Statistical Power ($power[-1]$)", fontsize=11, labelpad=8)
    ax.set_xlabel("Anytime Test Type", fontsize=11, labelpad=8)
    
    # Enforce strict axis ticks sequence mapping back to the targeted tests labels
    ax.set_xticks(x_base_indices)
    ax.set_xticklabels(ordered_display_names, rotation=30, ha='right', fontsize=9)
    ax.set_xlim(-0.5, num_tests - 0.5)
    ax.set_ylim(-0.02, 1.02)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    plt.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)

    # Use your module scatter helper for legendary formatting layout adjustments
    apply_scatter_legend(ax)
    
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)
    plt.savefig(
        f"{plot_utils.SAVE_PATH}/merged_selection_end_power_n{n}.png", 
        bbox_inches='tight'
    )
    print("Unified terminal end power plot completely rendered.")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
        
    # Programmatically combine the layers without manual pairing entries
    evaluation_queue = plot_utils.merge_selection_kinds(base_non_bandit_modes, bandit_allocation_mechanisms, active_bandit_bounds)

    plot_utils.SELECTION_PALETTE = plot_utils.generate_minimal_palette_map(plot_utils.SELECTION_GROUPS, evaluation_queue)

    # Call the new merged visual processing graph compilation pipeline function
    build_and_plot_merged_end_power(evaluation_queue)