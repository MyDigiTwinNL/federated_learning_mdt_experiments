import os,sys
import glob
import pandas as pd
import numpy as np

from sklearn.calibration import calibration_curve
from lifelines import KaplanMeierFitter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Specify the directory containing your CSV files
current_dir = os.path.dirname(os.path.abspath(__file__))
daft_dir = os.path.join(current_dir, 'daft')
calib_table_dir = os.path.join(daft_dir, 'calib_table')

calib_plots_dir = os.path.join(daft_dir, 'calibration_plots')

if not os.path.exists(calib_plots_dir):
    os.makedirs(calib_plots_dir)

category = "cacs0"
category_dir = os.path.join(calib_table_dir, category)

calib_all_methods = {
    "folder_names": [
        "cox_trf",
        "deepsurv",
        "cox",
        "cox_computed",
        "dl_computed",
        "fullct",
        "heart",
        "concat5fc"
    ],
    "methods_names": [
        r"$\mathrm{Cox}_{\mathrm{trf}}$",
        r"$\mathrm{DL}_{\mathrm{trf}}$",
        r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{semi}}$",
        r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{auto}}$",
        r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{auto}}$",
        r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{fullCT}}$",
        r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{heart}}$",
        r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{CAC}}$"
    ],
    "colors": []
}

# Tab10 color palette
tab10 = plt.cm.tab10.colors

# Manual assignment using reordering
# Reserve indices 2 (green), 0 (blue), 3 (red)
color_mapping = {
    r"$\mathrm{DL}_{\mathrm{trf}}$": tab10[2],           # green
    r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{auto}}$": tab10[0],  # blue
    r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{CAC}}$": tab10[3]     # red
}

# Use remaining colors for the rest (avoiding used ones: 0, 2, 3)
remaining_colors = [tab10[i] for i in range(10) if i not in [0, 2, 3]]
remaining_idx = 0

# Final ordered color list for the bar chart
for label in calib_all_methods["methods_names"]:
    if label in color_mapping:
        calib_all_methods["colors"].append(color_mapping[label])
    else:
        calib_all_methods["colors"].append(remaining_colors[remaining_idx])
        remaining_idx += 1

# Mapping of methods to tab10 color indices and approximate colors (following 'labels' order)
# ------------------------------------------------------------------------------------------
# Method                                          | tab10 Index | Approx. Color
# ------------------------------------------------------------------------------------------
# r"$\mathrm{Cox}_{\mathrm{trf}}$"                | tab10[1]    | orange
# r"$\mathrm{DL}_{\mathrm{trf}}$"                 | tab10[2]    | green
# r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{semi}}$"| tab10[4]    | purple
# r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{auto}}$"| tab10[0]    | blue
# r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{auto}}$" | tab10[5]    | brown
# r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{fullCT}}$"| tab10[6]   | pink
# r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{heart}}$"| tab10[7]    | gray
# r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{CAC}}$"  | tab10[3]    | red



# One curve per method (mean of 10 runs)
# ±1 std shaded area for uncertainty

###########################################

# Parameters
t_star = 3650  # 10 years
# t_star = 1825  # 5 years
n_bins = 10
min_bin_size = 5

# bin_edges = np.linspace(0, 1, n_bins + 1)
# bin_edges = np.array([0.00, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.5, 0.7, 0.9, 1.0])

# Plot
plt.figure(figsize=(5, 5))

# all_predicted_risks = []

# for folder in calib_all_methods["folder_names"]:
#     folder_path = os.path.join(category_dir, folder)
#     csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

#     for csv_file in csv_files:
#         df = pd.read_csv(csv_file)
#         df = df.dropna(subset=['event', 'time', 'output'])
#         df['event'] = df['event'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
#         df['event_at_t_star'] = ((df['event'] == 1) & (df['time'] <= t_star)).astype(int)

#         kmf = KaplanMeierFitter()
#         kmf.fit(df['time'], df['event'])
#         S0_t_star = kmf.predict(t_star)

#         df['pred_risk'] = 1 - (S0_t_star ** df['output'])

#         all_predicted_risks.extend(df['pred_risk'].values)

# # Step 2: define global bin edges using quantiles
# global_bin_edges = np.quantile(all_predicted_risks, q=np.linspace(0, 1, n_bins + 1))

calib_all_methods["folder_names"] = [
        "concat5fc"
    ]

for method_idx, folder in enumerate(calib_all_methods["folder_names"]):
    method_name = calib_all_methods["methods_names"][method_idx]
    method_color = calib_all_methods["colors"][method_idx]

    all_pred, all_obs = [], []

    folder_path = os.path.join(category_dir, folder)
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))


    for csv_file in csv_files:

        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['event', 'time', 'output'])

        df['event'] = df['event'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
        df['event_at_t_star'] = ((df['event'] == 1) & (df['time'] <= t_star)).astype(int)

        # df['event_at_t_star'] = ((df['time'] <= t_star)).astype(int)

        kmf = KaplanMeierFitter()
        kmf.fit(df['time'], event_observed=df['event'])
        S0_t_star = kmf.predict(t_star)

        df['pred_surv'] = S0_t_star ** df['output']
        df['pred_risk'] = 1 - df['pred_surv']

        # bins = pd.cut(df['pred_risk'], bins=global_bin_edges, labels=False, include_lowest=True)

        # bins = pd.cut(df['pred_risk'], bins=bin_edges, labels=False, include_lowest=True)


        # Bin using quantiles
        try:
            bins = pd.qcut(df['pred_risk'], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            bins = pd.cut(df['pred_risk'], bins=n_bins, labels=False, include_lowest=True)

        # Debug: Print bin counts
        print(f"[{method_name}] {os.path.basename(csv_file)} - Bin counts: {np.bincount(bins.dropna().astype(int))}")



        pred_means, obs_means = [], []
        for b in range(n_bins):
            group = df[bins == b]
            if len(group) > 0:
                pred_means.append(group['pred_risk'].mean())
                obs_means.append(group['event_at_t_star'].mean())
            else:
                pred_means.append(np.nan)
                obs_means.append(np.nan)

        all_pred.append(pred_means)
        all_obs.append(obs_means)

    all_pred = np.array(all_pred)
    all_obs = np.array(all_obs)

    mean_pred = np.nanmean(all_pred, axis=0)
    mean_obs = np.nanmean(all_obs, axis=0)
    std_obs = np.nanstd(all_obs, axis=0)

    plt.plot(mean_obs, mean_pred, marker='o', color=method_color, label=method_name)
    plt.fill_betweenx(mean_pred, mean_obs - std_obs, mean_obs + std_obs,
                     color=method_color, alpha=0.2)

# Perfect calibration reference
# plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
plt.plot([0, 0.5], [0, 0.5], '--', color='gray', label='Perfect calibration')

# Labels and legend
plt.xlabel("Observed event rate")
plt.ylabel("Mean predicted risk")
# plt.title("Calibration Curves Across 10 Methods")
plt.legend(fontsize=11)
plt.ylim(0, 0.5)
plt.xlim(0, 0.5)
# plt.grid(True)
# Final layout
plt.tight_layout()

plt.savefig(os.path.join(calib_plots_dir, "calibration_lines_%s.png" %category), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(calib_plots_dir, "calibration_lines_%s.eps" %category), format='eps', bbox_inches='tight')  # no transparency


