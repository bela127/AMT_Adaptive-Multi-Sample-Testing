import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs

from amt.configuration import Config
import plot_utils

# --- SIMULATION CONFIG PROPERTIES ---
n, m = 20, 1
max_iterations = 2000
common_p, p_diff = 0.5, 0.1
reps = 2500
sel_mode = "beta.med"

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)
    
    # Pfad für die p_and_p_diff Experimente setzen
    load_dir = "./exp_results/test_res/p_and_p_diff"
    
    # Alle Testmodi aus der zentralen Test-Gruppe laden
    test_modes = list(plot_utils.TEST_GROUPS.keys())
    
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    plotted_any = False
    
    # Styling-Zähler initialisieren
    group_counters = {g_key: 0 for g_key in plot_utils.TEST_PALETTE.keys()}
    line_index = 0

    for test_mode in test_modes:
        # Konfiguration für p=0.5 und p_diff=0.1 aufbauen
        dummy_conf = Config(
            n=n, m=m, sample_size=max_iterations, initial_size=10, reps=reps,
            common_p=common_p, p_diff=p_diff, hyp=m, 
            selection_mode=sel_mode, test_mode=test_mode
        )
        
        file_path = f"{load_dir}/reject_{dummy_conf.get_test_name()}.npy"
        
        processed = plot_utils.load_and_process_results(file_path, test_mode)
        if processed is None:
            print(f"Skipping {test_mode}: File not found.")
            continue
            
        test_decision, num_reps, num_iters = processed
        
        # Power-Verlauf über alle Iterationen t berechnen
        power_over_time = np.sum(test_decision, axis=0) / num_reps
        
        # Metadaten & konsistentes Line-Style-Dictionary laden
        display_label, group_key = plot_utils.resolve_test_metadata(test_mode)
        
        style = plot_utils.get_line_style(
            group_key=group_key, 
            group_counter=group_counters.get(group_key, 0), 
            line_index=line_index, 
            color_lookup_table=plot_utils.TEST_PALETTE,
            marker_stride=250  
        )
        
        if group_key in group_counters:
            group_counters[group_key] += 1
        line_index += 1
        
        # Zeitreihen-Kurve plotten
        ax.plot(
            np.arange(0, num_iters),
            power_over_time,
            label=display_label,
            **style
        )
        plotted_any = True

    if plotted_any:
        selection_display_name = plot_utils.resolve_selection_metadata(sel_mode)[0]
        
        ax.set_title(
            f"Power Over Time Across Tests with Selection {selection_display_name}\n"
            f"($K={n}$, $M={m}$, $p={common_p}$, $\\Delta p={p_diff}$)", 
            fontsize=13, fontweight='bold', pad=15
        )
        ax.set_xlabel("Sample Time ($t$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max_iterations)
        
        plot_utils.apply_standard_legend(ax)
        
        save_name = f"power_over_time_all_tests_pdiff_{p_diff}_n{n}"
        plot_utils.save_fig(fig, save_name, plot_utils.SAVE_PATH, "svg")
        plot_utils.save_fig(fig, save_name, plot_utils.SAVE_PATH, "png")
        plt.close()
        print(f"Erfolgreich gespeichert unter: {save_name}")