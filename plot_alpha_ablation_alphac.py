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
p_diff = 0.0  # Under H0, p_diff is 0
target_alpha = 0.05

# m = 0 represents the Null Hypothesis (Type-I Error / FPR)
m = 0  
alpha_c_values = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    makedirs(plot_utils.SAVE_PATH, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    plotted_any = False  
    
    # Sequential palette for sweeping numerical hyperparameter values
    colors = sns.color_palette("cubehelix", n_colors=len(alpha_c_values))
    
    for idx, alpha_c in enumerate(alpha_c_values):
        load_dir = f"./exp_results/test_res/hyperpar_alpha/alpha{alpha_c}"
        
        dummy_conf = Config(
            n=n, 
            m=m, 
            sample_size=max_iterations, 
            initial_size=initial_size, 
            reps=reps,
            common_p=common_p, 
            p_diff=p_diff, 
            hyp=m, 
            selection_mode=sel_mode, 
            test_mode=test_mode,
            coin_weights="posdif",
            alpha_c=alpha_c
        )
        
        file_path = f"{load_dir}/reject_{dummy_conf.get_test_name()}.npy"
        
        processed = plot_utils.load_and_process_results(file_path, test_mode)
        if processed is None:
            continue
            
        test_decision, num_reps, num_iters = processed
        type_i_error = np.sum(test_decision, axis=0) / num_reps
        
        # Line style cycle for black-and-white accessibility
        linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]
        style_idx = idx % len(linestyles)
        
        ax.plot(
            np.arange(0, num_iters), 
            type_i_error, 
            label=rf"$\alpha_c = {alpha_c}$", 
            color=colors[idx],
            linestyle=linestyles[style_idx],
            linewidth=2.0
        )
        plotted_any = True

    if plotted_any:
        # Add the red reference threshold for nominal target alpha
        ax.axhline(
            y=target_alpha, 
            color='#b71c1c', 
            linestyle='-', 
            linewidth=2.0, 
            alpha=0.8, 
            zorder=10, 
            label=f"Nominal Alpha Threshold ($\\alpha = {target_alpha}$)"
        )
        
        ax.set_title(
            f"Ablation Study: Empirical Type-I Error Rates Under Global Null ($m=0$)\n"
            f"(Test: `{test_mode}`, Selection: `{sel_mode}`, $N={n}$, $H_0: p_i = {common_p}$)", 
            fontsize=13, fontweight='bold', pad=15
        )
        ax.set_xlabel("Sequential Sample Observations ($t$)", fontsize=11, labelpad=8)
        ax.set_ylabel("False Positive Rate (Empirical $\\alpha$)", fontsize=11, labelpad=8)
        
        # Zoom Y-axis focus to target_alpha context
        ax.set_ylim(-0.005, target_alpha * 2.5)
        ax.set_xlim(0, max_iterations)
        
        plot_utils.apply_standard_legend(ax)
        
        save_name = f"ablation_alpha_c_error_m0_n{n}_{sel_mode}_{test_mode.replace('.', '_')}.png"
        plt.savefig(f"{plot_utils.SAVE_PATH}/{save_name}", bbox_inches='tight')
        plt.close()
        print(f"Saved Type-I Error curve plot: {save_name}")