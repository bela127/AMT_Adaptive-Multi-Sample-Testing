# plot_utils.py
from os import makedirs
import os
import sys
import numpy as np
import seaborn as sns

import matplotlib as mpl
import matplotlib.lines as mlines

mpl.rcParams['text.usetex'] = True
# Replace default preamble to omit type1cm
mpl.rcParams['text.latex.preamble'] = r'''
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
'''

LOAD_PATH = "./test_res"
SAVE_PATH = "./plot_res"

FIXED_BATCH_MODES = [
    "bayesian.beta", 
    "bayesian.beta.loo", 
    "betabinom.pmf", 
    "betabinom.extreme.pmf", 
    "betabinom.isolated.extreme",
    "chi2",
    "kw"
]

# =================================================================
# METADATA DICTIONARIES (DECOUPLED SEMANTICS)
# =================================================================

TEST_GROUPS = {
    "betabinom.pmf":               (r"\textbf{Beta-Binom LOO (Ours)}",     "betabinom_counts"),
    "bayesian.e.variable":         ("Beta-Bayesian Factor E-Var", "e_vars"),
#    "one.vs.rest.beta.mixture":    ("One-vs-Rest Beta Mixture E-Var", "e_vars"),
#    "beta.mixture.infinite":       ("EBP Beta Mixture E-Var",    "e_vars"),
#    "ebp.mixture.infinite":        ("EBP Normal Mixture E-Var",  "e_vars"),
    "betting.e.variable":          ("Betting E-Var",             "e_vars"),
#    "betabinom.extreme.pmf":       ("Beta-Binom Extreme LOO",    "betabinom_counts"),
#    "betabinom.isolated.extreme":  ("Beta-Binom Isolated Pair",  "betabinom_counts"),
#    "bayesian.beta":               ("Bayesian Beta Interval",    "bayesian_beta"),
#    "bayesian.beta.loo":           ("Bayesian Beta LOO",         "bayesian_beta"),
    "chernoff.horizon":            ("Chernoff (Horizon)",        "chernoff"),
    "chernoff.infinite":           ("Chernoff (Infinite)",       "chernoff"),
    "hoeffding.variance.horizon":  ("Hoeffding (Var, Horizon)",  "hoeffding"),
    "hoeffding.empiric.horizon":   ("Hoeffding (Emp, Horizon)",  "hoeffding"),
    "hoeffding.variance.infinite": ("Hoeffding (Var, Infinite)", "hoeffding"),
    "hoeffding.empiric.infinite":  ("Hoeffding (Emp, Infinite)", "hoeffding"),
    "bernstein.variance.horizon":  ("Bernstein (Var, Horizon)",  "bernstein"),
    "bernstein.empiric.horizon":   ("Bernstein (Emp, Horizon)",  "bernstein"),
    "bernstein.variance.infinite": ("Bernstein (Var, Infinite)", "bernstein"),
    "bernstein.empiric.infinite":  ("Bernstein (Emp, Infinite)", "bernstein"),
    "lil.variance":                ("LIL (Variance)",            "lil"),
    "lil.empiric":                 ("LIL (Empirical)",           "lil"),
    "kl.horizon":                  ("KL-UCB (Horizon)",          "kl"),
    "kl.infinite":                 ("KL-UCB (Infinite)",         "kl"),
#    "glrt.horizon":                ("GLRT (Horizon)",            "glrt"),
#    "glrt.infinite":               ("GLRT (Infinite)",           "glrt"),
    "chi2":               (r"$\chi^2$ Test",           "baseline"),
    "kw":               ("Kruskal-Wallis",           "baseline"),
}

SELECTION_GROUPS = {
#    "beta.max":      ("Beta Extreme",   "bayesian_beta"),
    "beta.med":      (r"\textbf{Beta Med (Ours)}",  "bayesian_beta"),
#    "evar":          ("LOO E-Var",  "e_vars"),
#    "rand":          ("Uniform Random",             "baseline"),
    "equal":         ("Space Filling",              "baseline"),
    "ts":            ("Thompson Sampling (TS)",          "ts"),
    "ts.5":          ("TS High Exploitation",          "ts"),
    "means":          ("Mean Diff. Gready",          "baseline"),
    "mean.slow":     ("Mean Diff. Vanishing Eps.",          "baseline"),
}

for b_kind, (label, group_key) in  TEST_GROUPS.items():
    selection_modes = {
        "bandit.max":    "Sel.",
        "bandit.ratio":  "Ratio"
    }
    for sel_mode, sel_label in selection_modes.items():
        SELECTION_GROUPS[f"{sel_mode}.{b_kind}"] = (f"{sel_label} {label}", group_key)

STYLE_CYCLE = [
    {"linestyle": "-",  "alpha": 1.0, "lw": 2.2, "marker": "o"}, 
    {"linestyle": "--", "alpha": 0.8, "lw": 1.8, "marker": "s"}, 
    {"linestyle": "-.", "alpha": 0.7, "lw": 1.5, "marker": "^"}, 
    {"linestyle": ":",  "alpha": 0.6, "lw": 1.5, "marker": "D"}, 
    {"linestyle": (0, (5, 10)), "alpha": 0.7, "lw": 1.8, "marker": "h"}
]

# =================================================================
# GENERAL PALETTE REUSABLE GENERATOR
# =================================================================

def generate_palette_map(source_dict, sns_palette_name="cubehelix"):
    """Extracts unique group strings and returns an active hexadecimal mapping."""
    unique_groups = []
    for _, g_key in source_dict.values():
        if g_key not in unique_groups:
            unique_groups.append(g_key)
    if "fallback" not in unique_groups:
        unique_groups.append("fallback")
    
    colors = sns.color_palette(sns_palette_name, len(unique_groups)).as_hex()
    return dict(zip(unique_groups, colors))

# Build explicit separate palettes so colors match their context properties
TEST_PALETTE = generate_palette_map(TEST_GROUPS, "cubehelix")
SELECTION_PALETTE = generate_palette_map(SELECTION_GROUPS, "cubehelix")

def merge_selection_kinds(non_bandit_sel, bandit_sel, bandit_bounds):
    selection_kinds = []
    # Add non-bandit allocation baselines
    for sel in non_bandit_sel:
        selection_kinds.append((sel, None))
        
    # Dynamically match every active bound function with tracking allocators
    for b_kind in bandit_bounds:
        for sel in bandit_sel:
            selection_kinds.append((sel, b_kind))

    return selection_kinds

def get_selection_key(selection_mode, bandit_kind=None):
    if bandit_kind is not None:
        selection_mode = f"{selection_mode}.{bandit_kind}"
    return selection_mode

def generate_minimal_palette_map(source_dict, selection_kinds, sns_palette_name="cubehelix"):
    """Extracts unique group strings and returns an active hexadecimal mapping."""
    unique_groups = []
    for sel_mode, b_kind in selection_kinds:
        _, g_key = source_dict[get_selection_key(sel_mode, b_kind)]
        if g_key not in unique_groups:
            unique_groups.append(g_key)
    if "fallback" not in unique_groups:
        unique_groups.append("fallback")
    
    colors = sns.color_palette(sns_palette_name, len(unique_groups)).as_hex()
    return dict(zip(unique_groups, colors))

# =================================================================
# METADATA RESOLUTION ENGINE
# =================================================================

def resolve_test_metadata(test_mode):
    if test_mode in TEST_GROUPS:
        return TEST_GROUPS[test_mode]
    return f"{test_mode.replace('.', ' ').title()} (Unmapped)", "fallback"

def resolve_selection_metadata(selection_mode, bandit_kind=None):
    if bandit_kind is not None:
        selection_mode = f"{selection_mode}.{bandit_kind}"
    if selection_mode in SELECTION_GROUPS:
        return SELECTION_GROUPS[selection_mode]
    return f"{selection_mode.replace('.', ' ').title()} (Unmapped)", "fallback"

# =================================================================
# GENERALIZED LINE STYLE GENERATOR
# =================================================================

def get_line_style(group_key, group_counter, line_index, color_lookup_table, marker_stride=250, group_offset = False):
    offset = 0
    if group_offset:
        for key in color_lookup_table:
            if key == group_key:
                break
            offset +=1
    current_style_idx = (offset + group_counter) % len(STYLE_CYCLE) 
    style_config = STYLE_CYCLE[current_style_idx].copy()
    
    marker_start_offset = (line_index * 40) % marker_stride
    style_config["markevery"] = (marker_start_offset, marker_stride) if marker_stride > 1 else 1
    style_config["color"] = color_lookup_table.get(group_key, color_lookup_table["fallback"])
    return style_config

# =================================================================
# STANDARD PLOTTING OVERLAYS & IO
# =================================================================

def apply_standard_legend(ax, bbox_to_anchor=(0.5, -0.13), ncol = 4):
    ax.legend(
        loc="upper center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        fontsize=9,
        frameon=True,
        facecolor='#fafafa',
        edgecolor='#e3e3e3',
        columnspacing=1.0,
        handletextpad=0.8,
        handlelength=3.0,   
        numpoints=1         
    )



def apply_scatter_legend(ax,
                         bbox_to_anchor=(0.5, -0.15),
                         title=None,
                         left_pad_pts=1,
                         ncol = 1,
                         alignment='left',
                         loc="lower left"
                        ):
    """
    Erstellt eine Legende mit explizitem Abstand VOR dem Marker-Symbol.
    """
    # 1. Bestehende Handles und Labels von den Axes abgreifen
    handles, labels = ax.get_legend_handles_labels()
    
    # 2. Erstelle für jeden Marker ein leeres Spacer-Element
    #    (Ein unsichtbarer Punkt mit fester Breite als Vorlauf)
    spacer = mlines.Line2D([], [], color='none', marker='', markersize=0)
    
    # 3. Verknüpfe den Spacer vor jeden echten Handle
    padded_handles = [(spacer, h) for h in handles]
    
    ax.legend(
        handles=padded_handles,
        labels=labels,
        title=title,
        alignment=alignment,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        fontsize=9,
        frameon=True,
        facecolor='#fafafa',
        edgecolor='#e3e3e3',
        columnspacing=1.8,
        handletextpad=0.4,
        handlelength=left_pad_pts,
        scatterpoints=1,
    )

def load_and_process_results(file_path, test_mode, annytime = True):
    try:
        raw_data = np.load(file_path)
        tests = np.squeeze(raw_data, axis=0) 
        num_reps, num_iters = tests.shape
        
        if test_mode in FIXED_BATCH_MODES or annytime == False:
            test_decision = tests.astype(bool)
        else:
            test_decision = np.maximum.accumulate(tests, axis=1).astype(bool)
            
        return test_decision, num_reps, num_iters
    except FileNotFoundError:
        print(f"Failed loading {file_path}.")
        return None


def save_fig(fig, name, path, format="svg"):
    loc = os.path.join(path,f"{name}.{format}")
    fig.savefig(loc, format=format, bbox_inches='tight', transparent="True", pad_inches=0)