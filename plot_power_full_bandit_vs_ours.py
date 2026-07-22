import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs

from amt.bandit_bounds import bandit_bounds
from amt.configuration import Config
import plot_utils

# --- SIMULATION CONFIG PROPERTIES ---
n, m = 20, 1
max_iterations = 2000
common_p, p_diff = 0.5, 0.1
reps = 2500

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    plotted_any = False
    
    # Extrahiere alle aktiven Banditen-Typen aus dem Repository
    active_bandit_bounds = list(bandit_bounds.keys())
    
    # Primitiver Aufbau der Test-Pipeline 
    # Format: (selection_mode, test_mode, bandit_kind, load_path)
    targeted_combinations = [
        # 1. Unser Ansatz
        ("beta.med", "betabinom.pmf", None, "./exp_results/test_res/non_bandit_selection"),
        # 2. Alternativ-Ansatz
        #("evar", "one.vs.rest.beta.mixture", None, "./exp_results/test_res/non_bandit_selection")
        ("beta.med", "bayesian.e.variable", None, "./exp_results/test_res/non_bandit_selection"),
        ("beta.med", "betting.e.variable", None, "./exp_results/test_res/non_bandit_selection"),
    ]
    
    # 3. Dynamisch die passenden Bandit-Paare hinzufügen
    for b_kind in active_bandit_bounds:
        targeted_combinations.append((
            "bandit.max",
            b_kind,       # Matching Test-Mode
            b_kind,       # Matching Bandit-Kind
            "./exp_results/test_res/matching_bandit_sel"
        ))

    # Erzeuge die adaptive Palette basierend auf allen vorkommenden Kombinationen
    evaluation_queue = [(sel, b) for sel, _, b, _ in targeted_combinations]

    # Initialisiere Styling-Zähler analog zu deiner originalen Render-Schleife
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0
    
    for sel_mode, test_mode, b_kind, load_dir in targeted_combinations:
        # Automatische Auflösung des Display-Labels und des Gruppen-Schlüssels
        display_label, group_key = plot_utils.resolve_test_metadata(test_mode)
        
        dummy_conf = Config(
            n=n, m=m, sample_size=max_iterations, initial_size=10, reps=reps,
            common_p=common_p, p_diff=p_diff, hyp=m, 
            selection_mode=sel_mode, test_mode=test_mode
        )
        
        if b_kind is not None:
            dummy_conf.bandit_kind = b_kind
            
        file_path = f"{load_dir}/reject_{dummy_conf.get_test_name()}.npy"
        
        processed = plot_utils.load_and_process_results(file_path, test_mode)
        if processed is None:
            print(f"Skipping {display_label}: File not found.")
            continue
            
        test_decision, num_reps, num_iters = processed
        power_over_time = np.sum(test_decision, axis=0) / num_reps
        
        # Generiere das exakte Style-Dictionary aus deinen plot_utils
        style = plot_utils.get_line_style(
            group_key=group_key, 
            group_counter=group_counters.get(group_key, 0), 
            line_index=line_index, 
            color_lookup_table=plot_utils.TEST_PALETTE,
            marker_stride=250  
        )
        
        # Inkrementiere die Zähler für die nachfolgenden Linien
        if group_key in group_counters:
            group_counters[group_key] += 1
        line_index += 1
        
        # Plot nutzt ausschließlich das ungefilterte style-Dictionary
        ax.plot(
            np.arange(0, num_iters),
            power_over_time,
            label=display_label,
            **style
        )
        plotted_any = True

    if plotted_any:
        ax.set_title(
            f"Power of Bandits with Matching Bandit Selection vs. Adaptive Approaches with Selection {plot_utils.resolve_selection_metadata("beta.med")[0]}\n"
            f"($K={n}$, $M={m}$, $p={common_p}$, $\\Delta p={p_diff}$)", 
            fontsize=13, fontweight='bold', pad=15
        )
        ax.set_xlabel("Sample Time ($t$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max_iterations)
        
        plot_utils.apply_standard_legend(ax)
        
        save_name = f"power_full_bandit_vs_ours_n{n}"
        plot_utils.save_fig(fig, save_name, plot_utils.SAVE_PATH, "svg")
        plot_utils.save_fig(fig, save_name, plot_utils.SAVE_PATH, "png")
        plt.close()
        print(f"Erfolgreich gespeichert unter: {save_name}")