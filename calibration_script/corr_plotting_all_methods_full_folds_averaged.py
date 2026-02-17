

import os,sys
import glob
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Specify the directory containing your CSV files
current_dir = os.path.dirname(os.path.abspath(__file__))
daft_dir = os.path.join(current_dir, 'daft')
calib_table_dir = os.path.join(daft_dir, 'calib_table')

calib_plots_dir = os.path.join(daft_dir, 'calib_plots')


if not os.path.exists(calib_plots_dir):
    os.makedirs(calib_plots_dir)


category = "cacs0"
category_dir = os.path.join(calib_table_dir, category)
##########################################################


# # Optional: Only include files ending with .csv
# csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Create dictionary with keys: folder_names, methods_names, averaged_value

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
    "averaged_m_value": [

    ],
    "std_m_value": [

    ],

    "averaged_b_value": [

    ],
    "std_b_value": [

    ],
    "colors": []
}


for folder in calib_all_methods["folder_names"]:
    folder_path = os.path.join(category_dir, folder)
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    m_list = []
    b_list = []
    for csv_file in csv_files:
        # print ("csv_file", csv_file)

        merged_df = pd.read_csv(csv_file)
        merged_df = merged_df[merged_df["output"]<=10]
        


        nn_true_df = merged_df[merged_df["event"]==True]

        true_output = nn_true_df["output"].values
        nn_true_df["true_output_normalized"] = (true_output - true_output.min()) / (true_output.max() - true_output.min())


        m, b = np.polyfit(merged_df["time"], merged_df["normalized_output"], 1)
        m_list.append (m)
        b_list.append (b)


    m_avg = np.mean(m_list)
    m_std = np.std(m_list)
    b_avg = np.mean(b_list)
    b_std = np.std(b_list)

    calib_all_methods["averaged_m_value"].append(m_avg)
    calib_all_methods["std_m_value"].append(m_std)
    calib_all_methods["averaged_b_value"].append(b_avg)
    calib_all_methods["std_b_value"].append(b_std)


 


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

print (calib_all_methods)  

plt.figure(figsize=(4,4))
# matplotlib.use('Agg')
ax = plt.gca() # Get a matplotlib's axes instance
plt.xlabel("Time-to-event (Days)", fontsize=11)
plt.ylabel("Normalized partial hazard", fontsize=11)

# X_plot = np.linspace(-100,ax.get_xlim()[1],100)
X_plot = np.linspace(0, 3650, 50)

for i in range(len(calib_all_methods["methods_names"])):
    m = calib_all_methods["averaged_m_value"][i]
    # b = calib_all_methods["averaged_b_value"][i]
    b = 1.0
    std_m = calib_all_methods["std_m_value"][i]
    color = calib_all_methods["colors"][i]
    label = calib_all_methods["methods_names"][i]
    
    # Solid line: mean prediction
    Y_plot = m * X_plot + b
    ax.plot(X_plot, Y_plot, color=color, label=label, linewidth=2)

    # Shaded area: std band
    Y_lower = (m - std_m) * X_plot + b
    Y_upper = (m + std_m) * X_plot + b
    ax.fill_between(X_plot, Y_lower, Y_upper, color=color, alpha=0.1)

# Add legend
ax.legend(loc='lower left', fontsize=12, frameon=False)

# Final layout
plt.tight_layout()

plt.savefig(os.path.join(calib_plots_dir, "calibration_lines_%s.png" %category), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(calib_plots_dir, "calibration_lines_%s.eps" %category), format='eps', bbox_inches='tight')  # no transparency
plt.show()

