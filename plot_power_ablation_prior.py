import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os import makedirs
from amt.configuration import Config
import plot_utils

# --- EXPERIMENT CONFIG PROPERTIES ---
n = 20
sel_mode = "beta.med"
test_mode = "betabinom.pmf"
max_iterations = 2000
initial_size = 10
reps = 2500
common_p = 0.5
p_diff = 0.075  # Korrigiert: Konsistent 0.075 für Config und Berechnung

# m = 1 repräsentiert die Alternativhypothese (Power)
m = 1  

# Die exakte Prior-Liste aus deiner korrigierten Simulationsschleife
prior_values = [
    (4, 2),
    (((common_p * n + p_diff) / n * 3), ((common_p * n - p_diff) / n * 3)),
    (2, 2),
    (1, 1),
    (0.5, 0.5),
    (0.25, 0.25),
    (0, 0)
]

prior_label_lookup = {
    (4, 2): rf"$Beta({4:.2f}, {2:.2f})$  (Misspecified)",
    (((common_p * n + p_diff) / n * 3), ((common_p * n - p_diff) / n * 3)): rf"$Beta({((common_p * n + p_diff) / n * 3):.2f}, {((common_p * n - p_diff) / n * 3):.2f})$  (Historical)",
    (2, 2): rf"$Beta({2:.2f}, {2:.2f})$  (Skeptical Baseline)",
    (1, 1): rf"$Beta({1:.2f}, {1:.2f})$  (Uniform)",
    (0.5, 0.5): rf"$Beta({0.5:.2f}, {0.5:.2f})$  (Jeffreys)",
    (0.25, 0.25): rf"$Beta({0.25:.2f}, {0.25:.2f})$  (Perks)",
    (0, 0): rf"$Beta({0:.2f}, {0:.2f})$  (Haldane, improper)"
}

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False  
    
    # Sequenzielle Farbpalette für die unterschiedlichen Prior-Verteilungen
    colors = sns.color_palette("viridis", n_colors=len(prior_values))
    
    for idx, prior in enumerate(prior_values):
        # Pfad-Routing zieht die Daten exakt aus dem Ordner der Generierung
        load_dir = f"./exp_results/test_res/vari_prior/prior_{prior[0]:.2f}_{prior[1]:.2f}"
        
        dummy_conf = Config(
            n=n, 
            m=m, 
            hyp=m,
            sample_size=max_iterations, 
            initial_size=initial_size, 
            reps=reps,
            common_p=common_p, 
            p_diff=p_diff, 
            selection_mode=sel_mode, 
            test_mode=test_mode,
            prior=prior
        )
        
        file_path = f"{load_dir}/reject_{dummy_conf.get_test_name()}.npy"
        
        processed = plot_utils.load_and_process_results(file_path, test_mode)
        if processed is None:
            continue
            
        test_decision, num_reps, num_iters = processed
        power = np.sum(test_decision, axis=0) / num_reps
        
        # Erweiterter Linienstil-Zyklus zur eindeutigen optischen Trennung
        linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 5)), (0, (1, 1))]
        style_idx = idx % len(linestyles)
        
        label_text = prior_label_lookup[prior]
        
        ax.plot(
            np.arange(0, num_iters), 
            power, 
            label=label_text, 
            color=colors[idx],
            linestyle=linestyles[style_idx],
            linewidth=2.0
        )
        plotted_any = True

    if plotted_any:
        ax.set_title(
            f"Ablation Study: Empirical Power under Alternative Hypothesis $H_1$ ($m={m}$)\n"
            f"(Test: `{test_mode}`, Selection: `{sel_mode}`, $N={n}$, $\\Delta p={p_diff}$)", 
            fontsize=13, fontweight='bold', pad=15
        )
        ax.set_xlabel("Sequential Sample Observations ($t$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Statistical Power ($1 - \\beta$)", fontsize=11, labelpad=8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max_iterations)
        
        plot_utils.apply_standard_legend(ax)
        
        save_name = f"ablation_prior_power_m1_n{n}_{sel_mode}_{test_mode.replace('.', '_')}.png"
        plt.savefig(f"{plot_utils.SAVE_PATH}/{save_name}", bbox_inches='tight')
        plt.close()
        print(f"Saved prior ablation power curve plot: {save_name}")