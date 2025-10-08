import re
import numpy as np

from amt.configuration import Config
from amt.utils import translate_keys

code = r'''
\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{adjustbox}

% Define a new command '\data' to act as a placeholder.
% It takes one argument (the variable name) and typesets "XX".
\newcommand{\data}[1]{XX}

\begin{document}

\begin{table*}[htbp]
\centering
\caption{Power and Gain of Multi-Sample Tests with Adaptive vs Non-Adaptive Selection Strategies at $\delta p = 0.05$}
\label{tab:multisample_regrouped}
\begin{adjustbox}{width=\textwidth,center}
\begin{tabular}{@{}llcccccccccc@{}}
\toprule
& & \multicolumn{2}{c}{\textbf{Mean}} & \multicolumn{2}{c}{\textbf{Chi-squared}} & \multicolumn{2}{c}{\textbf{Kruskal-Wallis}} & \multicolumn{2}{c}{\textbf{Beta}} & \multicolumn{2}{c}{\textbf{Beta-Binomial}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12}
\textbf{N / I} & \textbf{Method} & \textbf{Power} & \textbf{Gain (\%)} & \textbf{Power} & \textbf{Gain (\%)} & \textbf{Power} & \textbf{Gain (\%)} & \textbf{Power} & \textbf{Gain (\%)} & \textbf{Power} & \textbf{Gain (\%)} \\
\midrule

% --- Block for N=10, Size=2000 ---
\multirow{7}{*}{\textbf{10 / 1000}} & Equal & \data{power_mean_equal_10_2000} & -- & \data{power_chi2_equal_10_2000} & -- & \data{power_kw_equal_10_2000} & -- & \data{power_beta_equal_10_2000} & -- & \data{power_betabinom_equal_10_2000} & -- \\
\cmidrule(l){2-12}
& TS5 & \data{power_mean_ts5_10_2000} & \data{gain_mean_ts5_10_2000} & \data{power_chi2_ts5_10_2000} & \data{gain_chi2_ts5_10_2000} & \data{power_kw_ts5_10_2000} & \data{gain_kw_ts5_10_2000} & \data{power_beta_ts5_10_2000} & \data{gain_beta_ts5_10_2000} & \data{power_betabinom_ts5_10_2000} & \data{gain_betabinom_ts5_10_2000} \\
& TS & \data{power_mean_ts_10_2000} & \data{gain_mean_ts_10_2000} & \data{power_chi2_ts_10_2000} & \data{gain_chi2_ts_10_2000} & \data{power_kw_ts_10_2000} & \data{gain_kw_ts_10_2000} & \data{power_beta_ts_10_2000} & \data{gain_beta_ts_10_2000} & \data{power_betabinom_ts_10_2000} & \data{gain_betabinom_ts_10_2000} \\
& Beta & \data{power_mean_beta_10_2000} & \data{gain_mean_beta_10_2000} & \data{power_chi2_beta_10_2000} & \data{gain_chi2_beta_10_2000} & \data{power_kw_beta_10_2000} & \data{gain_kw_beta_10_2000} & \data{power_beta_beta_10_2000} & \data{gain_beta_beta_10_2000} & \data{power_betabinom_beta_10_2000} & \data{gain_betabinom_beta_10_2000} \\
& Means & \data{power_mean_means_10_2000} & \data{gain_mean_means_10_2000} & \data{power_chi2_means_10_2000} & \data{gain_chi2_means_10_2000} & \data{power_kw_means_10_2000} & \data{gain_kw_means_10_2000} & \data{power_beta_means_10_2000} & \data{gain_beta_means_10_2000} & \data{power_betabinom_means_10_2000} & \data{gain_betabinom_means_10_2000} \\
& Mean Slow & \data{power_mean_meanslow_10_2000} & \data{gain_mean_meanslow_10_2000} & \data{power_chi2_meanslow_10_2000} & \data{gain_chi2_meanslow_10_2000} & \data{power_kw_meanslow_10_2000} & \data{gain_kw_meanslow_10_2000} & \data{power_beta_meanslow_10_2000} & \data{gain_beta_meanslow_10_2000} & \data{power_betabinom_meanslow_10_2000} & \data{gain_betabinom_meanslow_10_2000} \\
\midrule

% --- Block for N=15, Size=2000 ---
\multirow{7}{*}{\textbf{15 / 1000}} & Equal & \data{power_mean_equal_15_2000} & -- & \data{power_chi2_equal_15_2000} & -- & \data{power_kw_equal_15_2000} & -- & \data{power_beta_equal_15_2000} & -- & \data{power_betabinom_equal_15_2000} & -- \\
\cmidrule(l){2-12}
& TS5 & \data{power_mean_ts5_15_2000} & \data{gain_mean_ts5_15_2000} & \data{power_chi2_ts5_15_2000} & \data{gain_chi2_ts5_15_2000} & \data{power_kw_ts5_15_2000} & \data{gain_kw_ts5_15_2000} & \data{power_beta_ts5_15_2000} & \data{gain_beta_ts5_15_2000} & \data{power_betabinom_ts5_15_2000} & \data{gain_betabinom_ts5_15_2000} \\
& TS & \data{power_mean_ts_15_2000} & \data{gain_mean_ts_15_2000} & \data{power_chi2_ts_15_2000} & \data{gain_chi2_ts_15_2000} & \data{power_kw_ts_15_2000} & \data{gain_kw_ts_15_2000} & \data{power_beta_ts_15_2000} & \data{gain_beta_ts_15_2000} & \data{power_betabinom_ts_15_2000} & \data{gain_betabinom_ts_15_2000} \\
& Beta & \data{power_mean_beta_15_2000} & \data{gain_mean_beta_15_2000} & \data{power_chi2_beta_15_2000} & \data{gain_chi2_beta_15_2000} & \data{power_kw_beta_15_2000} & \data{gain_kw_beta_15_2000} & \data{power_beta_beta_15_2000} & \data{gain_beta_beta_15_2000} & \data{power_betabinom_beta_15_2000} & \data{gain_betabinom_beta_15_2000} \\
& Means & \data{power_mean_means_15_2000} & \data{gain_mean_means_15_2000} & \data{power_chi2_means_15_2000} & \data{gain_chi2_means_15_2000} & \data{power_kw_means_15_2000} & \data{gain_kw_means_15_2000} & \data{power_beta_means_15_2000} & \data{gain_beta_means_15_2000} & \data{power_betabinom_means_15_2000} & \data{gain_betabinom_means_15_2000} \\
& Mean Slow & \data{power_mean_meanslow_15_2000} & \data{gain_mean_meanslow_15_2000} & \data{power_chi2_meanslow_15_2000} & \data{gain_chi2_meanslow_15_2000} & \data{power_kw_meanslow_15_2000} & \data{gain_kw_meanslow_15_2000} & \data{power_beta_meanslow_15_2000} & \data{gain_beta_meanslow_15_2000} & \data{power_betabinom_meanslow_15_2000} & \data{gain_betabinom_meanslow_15_2000} \\
\midrule

% --- Block for N=20, Size=2000 ---
\multirow{7}{*}{\textbf{20 / 1000}} & Equal & \data{power_mean_equal_20_2000} & -- & \data{power_chi2_equal_20_2000} & -- & \data{power_kw_equal_20_2000} & -- & \data{power_beta_equal_20_2000} & -- & \data{power_betabinom_equal_20_2000} & -- \\
\cmidrule(l){2-12}
& TS5 & \data{power_mean_ts5_20_2000} & \data{gain_mean_ts5_20_2000} & \data{power_chi2_ts5_20_2000} & \data{gain_chi2_ts5_20_2000} & \data{power_kw_ts5_20_2000} & \data{gain_kw_ts5_20_2000} & \data{power_beta_ts5_20_2000} & \data{gain_beta_ts5_20_2000} & \data{power_betabinom_ts5_20_2000} & \data{gain_betabinom_ts5_20_2000} \\
& TS & \data{power_mean_ts_20_2000} & \data{gain_mean_ts_20_2000} & \data{power_chi2_ts_20_2000} & \data{gain_chi2_ts_20_2000} & \data{power_kw_ts_20_2000} & \data{gain_kw_ts_20_2000} & \data{power_beta_ts_20_2000} & \data{gain_beta_ts_20_2000} & \data{power_betabinom_ts_20_2000} & \data{gain_betabinom_ts_20_2000} \\
& Beta & \data{power_mean_beta_20_2000} & \data{gain_mean_beta_20_2000} & \data{power_chi2_beta_20_2000} & \data{gain_chi2_beta_20_2000} & \data{power_kw_beta_20_2000} & \data{gain_kw_beta_20_2000} & \data{power_beta_beta_20_2000} & \data{gain_beta_beta_20_2000} & \data{power_betabinom_beta_20_2000} & \data{gain_betabinom_beta_20_2000} \\
& Means & \data{power_mean_means_20_2000} & \data{gain_mean_means_20_2000} & \data{power_chi2_means_20_2000} & \data{gain_chi2_means_20_2000} & \data{power_kw_means_20_2000} & \data{gain_kw_means_20_2000} & \data{power_beta_means_20_2000} & \data{gain_beta_means_20_2000} & \data{power_betabinom_means_20_2000} & \data{gain_betabinom_means_20_2000} \\
& Mean Slow & \data{power_mean_meanslow_20_2000} & \data{gain_mean_meanslow_20_2000} & \data{power_chi2_meanslow_20_2000} & \data{gain_chi2_meanslow_20_2000} & \data{power_kw_meanslow_20_2000} & \data{gain_kw_meanslow_20_2000} & \data{power_beta_meanslow_20_2000} & \data{gain_beta_meanslow_20_2000} & \data{power_betabinom_meanslow_20_2000} & \data{gain_betabinom_meanslow_20_2000} \\
\midrule[1.5pt] % Thicker rule between different sizes


% --- Block for N=10, Size=3000 ---
\multirow{7}{*}{\textbf{10 / 1500}} & Equal & \data{power_mean_equal_10_3000} & -- & \data{power_chi2_equal_10_3000} & -- & \data{power_kw_equal_10_3000} & -- & \data{power_beta_equal_10_3000} & -- & \data{power_betabinom_equal_10_3000} & -- \\
\cmidrule(l){2-12}
& TS5 & \data{power_mean_ts5_10_3000} & \data{gain_mean_ts5_10_3000} & \data{power_chi2_ts5_10_3000} & \data{gain_chi2_ts5_10_3000} & \data{power_kw_ts5_10_3000} & \data{gain_kw_ts5_10_3000} & \data{power_beta_ts5_10_3000} & \data{gain_beta_ts5_10_3000} & \data{power_betabinom_ts5_10_3000} & \data{gain_betabinom_ts5_10_3000} \\
& TS & \data{power_mean_ts_10_3000} & \data{gain_mean_ts_10_3000} & \data{power_chi2_ts_10_3000} & \data{gain_chi2_ts_10_3000} & \data{power_kw_ts_10_3000} & \data{gain_kw_ts_10_3000} & \data{power_beta_ts_10_3000} & \data{gain_beta_ts_10_3000} & \data{power_betabinom_ts_10_3000} & \data{gain_betabinom_ts_10_3000} \\
& Beta & \data{power_mean_beta_10_3000} & \data{gain_mean_beta_10_3000} & \data{power_chi2_beta_10_3000} & \data{gain_chi2_beta_10_3000} & \data{power_kw_beta_10_3000} & \data{gain_kw_beta_10_3000} & \data{power_beta_beta_10_3000} & \data{gain_beta_beta_10_3000} & \data{power_betabinom_beta_10_3000} & \data{gain_betabinom_beta_10_3000} \\
& Means & \data{power_mean_means_10_3000} & \data{gain_mean_means_10_3000} & \data{power_chi2_means_10_3000} & \data{gain_chi2_means_10_3000} & \data{power_kw_means_10_3000} & \data{gain_kw_means_10_3000} & \data{power_beta_means_10_3000} & \data{gain_beta_means_10_3000} & \data{power_betabinom_means_10_3000} & \data{gain_betabinom_means_10_3000} \\
& Mean Slow & \data{power_mean_meanslow_10_3000} & \data{gain_mean_meanslow_10_3000} & \data{power_chi2_meanslow_10_3000} & \data{gain_chi2_meanslow_10_3000} & \data{power_kw_meanslow_10_3000} & \data{gain_kw_meanslow_10_3000} & \data{power_beta_meanslow_10_3000} & \data{gain_beta_meanslow_10_3000} & \data{power_betabinom_meanslow_10_3000} & \data{gain_betabinom_meanslow_10_3000} \\
\midrule

% --- Block for N=15, Size=3000 ---
\multirow{7}{*}{\textbf{15 / 1500}} & Equal & \data{power_mean_equal_15_3000} & -- & \data{power_chi2_equal_15_3000} & -- & \data{power_kw_equal_15_3000} & -- & \data{power_beta_equal_15_3000} & -- & \data{power_betabinom_equal_15_3000} & -- \\
\cmidrule(l){2-12}
& TS5 & \data{power_mean_ts5_15_3000} & \data{gain_mean_ts5_15_3000} & \data{power_chi2_ts5_15_3000} & \data{gain_chi2_ts5_15_3000} & \data{power_kw_ts5_15_3000} & \data{gain_kw_ts5_15_3000} & \data{power_beta_ts5_15_3000} & \data{gain_beta_ts5_15_3000} & \data{power_betabinom_ts5_15_3000} & \data{gain_betabinom_ts5_15_3000} \\
& TS & \data{power_mean_ts_15_3000} & \data{gain_mean_ts_15_3000} & \data{power_chi2_ts_15_3000} & \data{gain_chi2_ts_15_3000} & \data{power_kw_ts_15_3000} & \data{gain_kw_ts_15_3000} & \data{power_beta_ts_15_3000} & \data{gain_beta_ts_15_3000} & \data{power_betabinom_ts_15_3000} & \data{gain_betabinom_ts_15_3000} \\
& Beta & \data{power_mean_beta_15_3000} & \data{gain_mean_beta_15_3000} & \data{power_chi2_beta_15_3000} & \data{gain_chi2_beta_15_3000} & \data{power_kw_beta_15_3000} & \data{gain_kw_beta_15_3000} & \data{power_beta_beta_15_3000} & \data{gain_beta_beta_15_3000} & \data{power_betabinom_beta_15_3000} & \data{gain_betabinom_beta_15_3000} \\
& Means & \data{power_mean_means_15_3000} & \data{gain_mean_means_15_3000} & \data{power_chi2_means_15_3000} & \data{gain_chi2_means_15_3000} & \data{power_kw_means_15_3000} & \data{gain_kw_means_15_3000} & \data{power_beta_means_15_3000} & \data{gain_beta_means_15_3000} & \data{power_betabinom_means_15_3000} & \data{gain_betabinom_means_15_3000} \\
& Mean Slow & \data{power_mean_meanslow_15_3000} & \data{gain_mean_meanslow_15_3000} & \data{power_chi2_meanslow_15_3000} & \data{gain_chi2_meanslow_15_3000} & \data{power_kw_meanslow_15_3000} & \data{gain_kw_meanslow_15_3000} & \data{power_beta_meanslow_15_3000} & \data{gain_beta_meanslow_15_3000} & \data{power_betabinom_meanslow_15_3000} & \data{gain_betabinom_meanslow_15_3000} \\
\midrule

% --- Block for N=20, Size=3000 ---
\multirow{7}{*}{\textbf{20 / 1500}} & Equal & \data{power_mean_equal_20_3000} & -- & \data{power_chi2_equal_20_3000} & -- & \data{power_kw_equal_20_3000} & -- & \data{power_beta_equal_20_3000} & -- & \data{power_betabinom_equal_20_3000} & -- \\
\cmidrule(l){2-12}
& TS5 & \data{power_mean_ts5_20_3000} & \data{gain_mean_ts5_20_3000} & \data{power_chi2_ts5_20_3000} & \data{gain_chi2_ts5_20_3000} & \data{power_kw_ts5_20_3000} & \data{gain_kw_ts5_20_3000} & \data{power_beta_ts5_20_3000} & \data{gain_beta_ts5_20_3000} & \data{power_betabinom_ts5_20_3000} & \data{gain_betabinom_ts5_20_3000} \\
& TS & \data{power_mean_ts_20_3000} & \data{gain_mean_ts_20_3000} & \data{power_chi2_ts_20_3000} & \data{gain_chi2_ts_20_3000} & \data{power_kw_ts_20_3000} & \data{gain_kw_ts_20_3000} & \data{power_beta_ts_20_3000} & \data{gain_beta_ts_20_3000} & \data{power_betabinom_ts_20_3000} & \data{gain_betabinom_ts_20_3000} \\
& Beta & \data{power_mean_beta_20_3000} & \data{gain_mean_beta_20_3000} & \data{power_chi2_beta_20_3000} & \data{gain_chi2_beta_20_3000} & \data{power_kw_beta_20_3000} & \data{gain_kw_beta_20_3000} & \data{power_beta_beta_20_3000} & \data{gain_beta_beta_20_3000} & \data{power_betabinom_beta_20_3000} & \data{gain_betabinom_beta_20_3000} \\
& Means & \data{power_mean_means_20_3000} & \data{gain_mean_means_20_3000} & \data{power_chi2_means_20_3000} & \data{gain_chi2_means_20_3000} & \data{power_kw_means_20_3000} & \data{gain_kw_means_20_3000} & \data{power_beta_means_20_3000} & \data{gain_beta_means_20_3000} & \data{power_betabinom_means_20_3000} & \data{gain_betabinom_means_20_3000} \\
& Mean Slow & \data{power_mean_meanslow_20_3000} & \data{gain_mean_meanslow_20_3000} & \data{power_chi2_meanslow_20_3000} & \data{gain_chi2_meanslow_20_3000} & \data{power_kw_meanslow_20_3000} & \data{gain_kw_meanslow_20_3000} & \data{power_beta_meanslow_20_3000} & \data{gain_beta_meanslow_20_3000} & \data{power_betabinom_meanslow_20_3000} & \data{gain_betabinom_meanslow_20_3000} \\
\midrule[1.5pt]


% --- Block for N=10, Size=4000 ---
\multirow{7}{*}{\textbf{10 / 2000}} & Equal & \data{power_mean_equal_10_4000} & -- & \data{power_chi2_equal_10_4000} & -- & \data{power_kw_equal_10_4000} & -- & \data{power_beta_equal_10_4000} & -- & \data{power_betabinom_equal_10_4000} & -- \\
\cmidrule(l){2-12}
& TS5 & \data{power_mean_ts5_10_4000} & \data{gain_mean_ts5_10_4000} & \data{power_chi2_ts5_10_4000} & \data{gain_chi2_ts5_10_4000} & \data{power_kw_ts5_10_4000} & \data{gain_kw_ts5_10_4000} & \data{power_beta_ts5_10_4000} & \data{gain_beta_ts5_10_4000} & \data{power_betabinom_ts5_10_4000} & \data{gain_betabinom_ts5_10_4000} \\
& TS & \data{power_mean_ts_10_4000} & \data{gain_mean_ts_10_4000} & \data{power_chi2_ts_10_4000} & \data{gain_chi2_ts_10_4000} & \data{power_kw_ts_10_4000} & \data{gain_kw_ts_10_4000} & \data{power_beta_ts_10_4000} & \data{gain_beta_ts_10_4000} & \data{power_betabinom_ts_10_4000} & \data{gain_betabinom_ts_10_4000} \\
& Beta & \data{power_mean_beta_10_4000} & \data{gain_mean_beta_10_4000} & \data{power_chi2_beta_10_4000} & \data{gain_chi2_beta_10_4000} & \data{power_kw_beta_10_4000} & \data{gain_kw_beta_10_4000} & \data{power_beta_beta_10_4000} & \data{gain_beta_beta_10_4000} & \data{power_betabinom_beta_10_4000} & \data{gain_betabinom_beta_10_4000} \\
& Means & \data{power_mean_means_10_4000} & \data{gain_mean_means_10_4000} & \data{power_chi2_means_10_4000} & \data{gain_chi2_means_10_4000} & \data{power_kw_means_10_4000} & \data{gain_kw_means_10_4000} & \data{power_beta_means_10_4000} & \data{gain_beta_means_10_4000} & \data{power_betabinom_means_10_4000} & \data{gain_betabinom_means_10_4000} \\
& Mean Slow & \data{power_mean_meanslow_10_4000} & \data{gain_mean_meanslow_10_4000} & \data{power_chi2_meanslow_10_4000} & \data{gain_chi2_meanslow_10_4000} & \data{power_kw_meanslow_10_4000} & \data{gain_kw_meanslow_10_4000} & \data{power_beta_meanslow_10_4000} & \data{gain_beta_meanslow_10_4000} & \data{power_betabinom_meanslow_10_4000} & \data{gain_betabinom_meanslow_10_4000} \\
\midrule

% --- Block for N=15, Size=4000 ---
\multirow{7}{*}{\textbf{15 / 2000}} & Equal & \data{power_mean_equal_15_4000} & -- & \data{power_chi2_equal_15_4000} & -- & \data{power_kw_equal_15_4000} & -- & \data{power_beta_equal_15_4000} & -- & \data{power_betabinom_equal_15_4000} & -- \\
\cmidrule(l){2-12}
& TS5 & \data{power_mean_ts5_15_4000} & \data{gain_mean_ts5_15_4000} & \data{power_chi2_ts5_15_4000} & \data{gain_chi2_ts5_15_4000} & \data{power_kw_ts5_15_4000} & \data{gain_kw_ts5_15_4000} & \data{power_beta_ts5_15_4000} & \data{gain_beta_ts5_15_4000} & \data{power_betabinom_ts5_15_4000} & \data{gain_betabinom_ts5_15_4000} \\
& TS & \data{power_mean_ts_15_4000} & \data{gain_mean_ts_15_4000} & \data{power_chi2_ts_15_4000} & \data{gain_chi2_ts_15_4000} & \data{power_kw_ts_15_4000} & \data{gain_kw_ts_15_4000} & \data{power_beta_ts_15_4000} & \data{gain_beta_ts_15_4000} & \data{power_betabinom_ts_15_4000} & \data{gain_betabinom_ts_15_4000} \\
& Beta & \data{power_mean_beta_15_4000} & \data{gain_mean_beta_15_4000} & \data{power_chi2_beta_15_4000} & \data{gain_chi2_beta_15_4000} & \data{power_kw_beta_15_4000} & \data{gain_kw_beta_15_4000} & \data{power_beta_beta_15_4000} & \data{gain_beta_beta_15_4000} & \data{power_betabinom_beta_15_4000} & \data{gain_betabinom_beta_15_4000} \\
& Means & \data{power_mean_means_15_4000} & \data{gain_mean_means_15_4000} & \data{power_chi2_means_15_4000} & \data{gain_chi2_means_15_4000} & \data{power_kw_means_15_4000} & \data{gain_kw_means_15_4000} & \data{power_beta_means_15_4000} & \data{gain_beta_means_15_4000} & \data{power_betabinom_means_15_4000} & \data{gain_betabinom_means_15_4000} \\
& Mean Slow & \data{power_mean_meanslow_15_4000} & \data{gain_mean_meanslow_15_4000} & \data{power_chi2_meanslow_15_4000} & \data{gain_chi2_meanslow_15_4000} & \data{power_kw_meanslow_15_4000} & \data{gain_kw_meanslow_15_4000} & \data{power_beta_meanslow_15_4000} & \data{gain_beta_meanslow_15_4000} & \data{power_betabinom_meanslow_15_4000} & \data{gain_betabinom_meanslow_15_4000} \\
\midrule

% --- Block for N=20, Size=4000 ---
\multirow{7}{*}{\textbf{20 / 2000}} & Equal & \data{power_mean_equal_20_4000} & -- & \data{power_chi2_equal_20_4000} & -- & \data{power_kw_equal_20_4000} & -- & \data{power_beta_equal_20_4000} & -- & \data{power_betabinom_equal_20_4000} & -- \\
\cmidrule(l){2-12}
& TS5 & \data{power_mean_ts5_20_4000} & \data{gain_mean_ts5_20_4000} & \data{power_chi2_ts5_20_4000} & \data{gain_chi2_ts5_20_4000} & \data{power_kw_ts5_20_4000} & \data{gain_kw_ts5_20_4000} & \data{power_beta_ts5_20_4000} & \data{gain_beta_ts5_20_4000} & \data{power_betabinom_ts5_20_4000} & \data{gain_betabinom_ts5_20_4000} \\
& TS & \data{power_mean_ts_20_4000} & \data{gain_mean_ts_20_4000} & \data{power_chi2_ts_20_4000} & \data{gain_chi2_ts_20_4000} & \data{power_kw_ts_20_4000} & \data{gain_kw_ts_20_4000} & \data{power_beta_ts_20_4000} & \data{gain_beta_ts_20_4000} & \data{power_betabinom_ts_20_4000} & \data{gain_betabinom_ts_20_4000} \\
& Beta & \data{power_mean_beta_20_4000} & \data{gain_mean_beta_20_4000} & \data{power_chi2_beta_20_4000} & \data{gain_chi2_beta_20_4000} & \data{power_kw_beta_20_4000} & \data{gain_kw_beta_20_4000} & \data{power_beta_beta_20_4000} & \data{gain_beta_beta_20_4000} & \data{power_betabinom_beta_20_4000} & \data{gain_betabinom_beta_20_4000} \\
& Means & \data{power_mean_means_20_4000} & \data{gain_mean_means_20_4000} & \data{power_chi2_means_20_4000} & \data{gain_chi2_means_20_4000} & \data{power_kw_means_20_4000} & \data{gain_kw_means_20_4000} & \data{power_beta_means_20_4000} & \data{gain_beta_means_20_4000} & \data{power_betabinom_means_20_4000} & \data{gain_betabinom_means_20_4000} \\
& Mean Slow & \data{power_mean_meanslow_20_4000} & \data{gain_mean_meanslow_20_4000} & \data{power_chi2_meanslow_20_4000} & \data{gain_chi2_meanslow_20_4000} & \data{power_kw_meanslow_20_4000} & \data{gain_kw_meanslow_20_4000} & \data{power_beta_meanslow_20_4000} & \data{gain_beta_meanslow_20_4000} & \data{power_betabinom_meanslow_20_4000} & \data{gain_betabinom_meanslow_20_4000} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}

\end{document}
'''

def get_keys_of_maximal_value(refs):
    max_value = max(refs.values())

    maximal_keys = []
    for key, value in refs.items():
        if value == max_value:
            maximal_keys.append(key)

    return maximal_keys

if __name__ == "__main__":
    load_path = "./test_res/H1"
    raw_data = {}
    ref_data = {}
    data = {}
    for n in [5, 10, 15, 20]:#
        for test_mode in ["mean", "chi2", "kw", "beta", "betabinom.comb"]:#"mean", "chi2", "kw", "beta", "betabinom.comb"
            for sel_mode in ["ts.5", "ts", "equal", "beta.med", "means", "mean.slow"]:# "ts.5", "ts", "equal", "beta", "means", "mean.slow"
                
                conf = Config(
                    n = n,
                    m = 1,
                    sample_size = 2000,
                    initial_size = 10,
                    reps = 5000,
                    common_p = 0.5,
                    p_diff = 0.05,
                    selection_mode = sel_mode,
                    test_mode = test_mode,
                    coin_weights = "posdif",
                )
                name = conf.get_test_name()
                power = np.load(f"{load_path}/power_{name}.npy")
                sizes = np.asarray([2000,3000,4000], dtype=int)
                index = (power.shape[1]-1) * sizes / (conf.sample_size * 2)
                for i, s in zip(index,sizes):
                    test, sel = translate_keys(test_mode, sel_mode)
                    table_index = f"power_{test}_{sel}_{n}_{s}"
                    raw_data[table_index] = power[1, int(i)]
                    data[table_index] = f"{power[1, int(i)]:.3f}"
                    if sel_mode == "equal":
                        ref_data[table_index] = power[1, int(i)]
                    print(f"Loaded {table_index} = {data[table_index]}")
        
    for key, value in raw_data.items():
        split_key = key.split("_")
        ref_index = f"power_{split_key[1]}_equal_{split_key[3]}_{split_key[4]}"
        table_index = f"gain_{split_key[1]}_{split_key[2]}_{split_key[3]}_{split_key[4]}"
        gain = (value - ref_data[ref_index]) / ref_data[ref_index] * 100
        data[table_index] = f"{gain:.0f}"
        print(f"Calculated {table_index} = {data[table_index]}")

    
    for n in [5, 10, 15, 20]:#
        for s in [2000, 3000, 4000]:
            for test_mode in ["mean", "chi2", "kw", "beta", "betabinom.comb"]:#"mean", "chi2", "kw", "beta", "betabinom.comb"
                refs = {}
                for sel_mode in ["ts.5", "ts", "equal", "beta.med", "means", "mean.slow"]:# "ts.5", "ts", "equal", "beta", "means", "mean.slow"     
                    test, sel = translate_keys(test_mode, sel_mode)
                    table_index = f"power_{test}_{sel}_{n}_{s}"
                    refs[table_index] = raw_data[table_index]
                
                keys = get_keys_of_maximal_value(refs)
                for key in keys:
                    split_key = key.split("_")
                    gain_index = f"gain_{split_key[1]}_{split_key[2]}_{split_key[3]}_{split_key[4]}"
                    data[key] = rf"\textbf{{{data[key]}}}"
                    data[gain_index] = rf"\textbf{{{data[gain_index]}}}"
                    print(f"Marked {key} as maximal with value {refs[key]}")

    def replace(matchobj):
        """
        Extracts the variable name from the match object, looks it up in a data directory,
        and returns its corresponding value for replacement.
        """
        variable_name = matchobj.group(1) 
        value = data.get(variable_name, "XX")
        if value == "XX":
            print(f"Warning: {variable_name} not found in data.")
        return value


    pattern = r'\\data\{(.*?)\}'

    result = re.sub(pattern, replace, code)

    print(result)


