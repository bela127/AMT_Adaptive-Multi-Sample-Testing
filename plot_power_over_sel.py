import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs

# Dynamic imports from your target structures
from amt.bandit_bounds import bandit_bounds
from amt.configuration import Config
import plot_utils

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    # --- SIMULATION CONFIG PROPERTIES ---
    n, m = 20, 1
    max_iterations = 2000
    common_p, p_diff = 0.5, 0.1
    reps = 2500
    
    # 1. Establish the evaluation baseline test mode
    target_test_mode = "chernoff.infinite"
    
    # 2. Extract active key arrays straight from the bandit_bounds repository
    active_bandit_bounds = list(bandit_bounds.keys())
    
    # 3. Define the core selection primitives to evaluate
    base_non_bandit_modes = ["rand", "ts", "beta.max", "beta.med"]
    bandit_allocation_mechanisms = ["bandit.max", "bandit.ratio"]
    
    # 4. Programmatically combine the layers without manual pairing entries
    evaluation_queue = []
    
    # Add non-bandit allocation baselines
    for mode in base_non_bandit_modes:
        evaluation_queue.append((mode, None))
        
    # Dynamically match every active bound function with your tracking allocators
    for b_kind in active_bandit_bounds:
        for alloc_mode in bandit_allocation_mechanisms:
            evaluation_queue.append((alloc_mode, b_kind))

    makedirs(plot_utils.SAVE_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False  
    
    # Use SELECTION_PALETTE tracking configurations
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
            
        file_path = f"{plot_utils.LOAD_PATH}/reject_{dummy_conf.get_test_name()}.npy"
        
        # Load profile matrix data cleanly
        processed = plot_utils.load_and_process_results(file_path, target_test_mode)
        if processed is None:
            # Silently drop unexecuted test/bound matrix configurations from the grid
            continue
            
        test_decision, num_reps, num_iters = processed
        power = np.sum(test_decision, axis=0) / num_reps
        
        # Resolve labels and colors using selection-specific lookups
        display_label, group_key = plot_utils.resolve_selection_metadata(sel_mode)
        
        # Append specific bound details to the display label if analyzing a bandit variant
        if b_kind:
            bound_label = b_kind.split('.')[0].title()  # e.g., 'Chernoff' or 'Kl'
            display_label = f"{display_label} ({bound_label} Bound)"
        else:
            display_label = display_label
            
        style = plot_utils.get_line_style(
            group_key=group_key, 
            group_counter=group_counters[group_key], 
            line_index=line_index, 
            color_lookup_table=plot_utils.SELECTION_PALETTE
        )
        
        group_counters[group_key] += 1 
        line_index += 1
        
        ax.plot(np.arange(0, num_iters), power, label=display_label, markersize=4, **style)
        plotted_any = True

    if plotted_any:
        ax.set_title(
            f"Selection Policy Power Profiles (Static Stopping Test: `{target_test_mode}`, $N={n}$)", 
            fontsize=13, fontweight='bold', pad=15
        )
        ax.set_xlabel("Observations ($t$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Statistical Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max_iterations)
        
        plot_utils.apply_standard_legend(ax, bbox_to_anchor=(0.5, -0.18))
        
        safe_test_out = target_test_mode.replace('.', '_')
        plt.savefig(
            f"{plot_utils.SAVE_PATH}/dynamic_selection_comparison_{safe_test_out}_n{n}.png", 
            bbox_inches='tight'
        )
        print("Dynamic selection plotting complete.")