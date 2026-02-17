
import os,sys
from pathlib import Path
sys.path.append('../')

# numpy_path = "/groups/umcg-lifelines/tmp02/projects/ov22_0581/python_pkgs/numpy-1.26.4.dist-info/"  # Adjust if needed
# pkg_dir = os.path.dirname(numpy_path)

# # Insert the package directory at the top
# sys.path.insert(0, pkg_dir)

sys.path.insert(0, '/groups/umcg-lifelines/tmp02/projects/ov22_0581/python_pkgs')
# sys.path.insert(0, '/groups/umcg-lifelines/tmp02/projects/ov22_0581/hmo/fedavg_lifelines_calibration/python_pkgs/scikit-survival/')

import numpy as np
# Print NumPy version
print(f"NumPy version: {np.__version__}")

import pandas 

print("Pandas version:", pandas.__version__)

import pandas as pd

import matplotlib.pyplot as plt
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator

import scipy.stats as st
from scipy.special import logit, expit


import glob
from sklearn.calibration import calibration_curve
from lifelines import KaplanMeierFitter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression




# dataset = "whas"
dataset = "lifelines"

# get path of current directory
current_path = Path(__file__).parent
current_dir = os.path.dirname(os.path.abspath(__file__))


# output_raw_dir = os.path.join(current_dir, "output_raw_lifelines")
output_raw_dir = os.path.join(current_dir, "output_raw_%s" %dataset)
calib_roc_dir = os.path.join(current_dir, "calibration_roc_plots")
if not os.path.exists(calib_roc_dir):
    os.makedirs(calib_roc_dir)


# Parameters

if dataset=="whas":
    t_star = 2000  # 5.5 years
else:
    t_star = 3650  # 10 years




n_bins = 10
min_bin_size = 5


# client_id = "0"
# client_id = "1"
# client_id = "2"
client_id = "glo"

update_iter_list = [20, 0]
method_list = ["FedAvg","Without aggregation"]

# Plot
plt.figure(figsize=(4, 4))

for method_index, update_iter in enumerate(update_iter_list):
    method_name = method_list[method_index]

    # colors = {"0":{"Without aggregation": "purple", "FedAvg": "purple"}, "1":{"Without aggregation": "red", "FedAvg": "red"}, "2":{"Without aggregation": "blue", "FedAvg": "blue"}, "glo":{"Without aggregation": "green", "FedAvg": "green"}}
    # linestyles = {"FedAvg": "-", "Without aggregation": "-"}

    # markers = {"FedAvg": "o", "Without aggregation": "o"}


    colors = {"0":{"Without aggregation": "blue", "FedAvg": "red"}, "1":{"Without aggregation": "blue", "FedAvg": "red"}, "2":{"Without aggregation": "blue", "FedAvg": "red"}, "glo":{"Without aggregation": "blue", "FedAvg": "red"}}
    linestyles = {"FedAvg": "-", "Without aggregation": "-"}

    markers = {"FedAvg": "o", "Without aggregation": "o"}

    all_pred, all_obs = [], []

    for fold_index in range(10):
        df_test_local_folder = os.path.join(output_raw_dir, str(fold_index))
        df_test_local_filepath = os.path.join(df_test_local_folder, "df_output_raw_%s_%s.csv" %(client_id,update_iter))
        df = pd.read_csv(df_test_local_filepath)

        

        if dataset=="whas":
            df = df[df["y"] < 2000.0]
        else:
            df = df[df["y"] < 3650.0]

        df.rename(columns={
            "pred_risk": "output",
            "y": "time",
            "e": "event"
        }, inplace=True)


        df = df.dropna(subset=['event', 'time', 'output'])
        df = df.dropna(subset=['time', 'event'])

        df['output'] = np.exp(df['output']) # log-risk for DeepSurv

        # df['event'] = df['event'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
        df['event_at_t_star'] = ((df['event'] == 1) & (df['time'] <= t_star)).astype(int)

        # df['event_at_t_star'] = ((df['time'] <= t_star)).astype(int)



        # if (folder == "concat5fc") or (folder == "heart") or (folder == "fullct"):
        #     df['output'] = np.log(df['output'])
        #     df['output'] = df['output']*4
        #     df['output'] = np.exp(df['output'])

        # print(df[df[['time', 'event']].isnull().any(axis=1)])

        kmf = KaplanMeierFitter()
        kmf.fit(df['time'], event_observed=df['event'])
        S0_t_star = kmf.predict(t_star)
        # print ("S0_t_star", S0_t_star)

        df['pred_surv'] = S0_t_star ** df['output']
        df['pred_risk'] = 1 - df['pred_surv']
        
        
        # # 2. Split into recalibration and evaluation sets
        # df_cal, df_eval = train_test_split(df, test_size=0.6, random_state=42)

        # # 3. Fit calibrator on 20%
        # calibrator = IsotonicRegression(out_of_bounds='clip')
        # calibrator.fit(df_cal['pred_risk'], df_cal['event_at_t_star'])

        # # 4. Apply to 80%
        # df_eval['pred_risk'] = calibrator.transform(df_eval['pred_risk'])

        # # 5. Use evaluation set for calibration analysis
        # df = df_eval.copy()
       

        # Bin using quantiles
        try:
            bins = pd.qcut(df['pred_risk'], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            bins = pd.cut(df['pred_risk'], bins=n_bins, labels=False, include_lowest=True)

        # Debug: Print bin counts
        print(f"[{method_name}] {os.path.basename(df_test_local_filepath)} - Bin counts: {np.bincount(bins.dropna().astype(int))}")


        pred_means, obs_means = [], []
        # for b in range(n_bins):
        #     group = df[bins == b]
        #     if len(group) > 0:
        #         pred_means.append(group['pred_risk'].mean())
        #         obs_means.append(group['event_at_t_star'].mean())
        #     else:
        #         pred_means.append(np.nan)
        #         obs_means.append(np.nan)


        for b in range(n_bins):
            group = df[bins == b]
            if len(group) >= min_bin_size:
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
    std_pred = np.nanstd(all_pred, axis=0)
    
    mean_obs = np.nanmean(all_obs, axis=0)
    std_obs = np.nanstd(all_obs, axis=0)
    
    valid = ~np.isnan(mean_pred) & ~np.isnan(mean_obs)
    mean_pred = mean_pred[valid]
    mean_obs = mean_obs[valid]
    std_pred = std_pred[valid]

    # window = 2  # or 3
    # mean_pred = uniform_filter1d(mean_pred, size=window, mode='nearest')
    # mean_obs = uniform_filter1d(mean_obs, size=window, mode='nearest')
    # std_pred = uniform_filter1d(std_pred, size=window, mode='nearest')

    # plt.plot(mean_obs, mean_pred, marker='o', color=method_color, label=method_name)
    # # plt.fill_betweenx(mean_pred, mean_obs - std_obs, mean_obs + std_obs,
    # #                  color=method_color, alpha=0.1)

    if method_name == "FedAvg":
        zorder_value = 10
    else:
        zorder_value = 1



    "FedAvg","Without aggregation"

    plt.plot(mean_obs, mean_pred, color=colors[client_id][method_name], linestyle=linestyles[method_name], marker = markers[method_name],  label=method_name, alpha= 0.7, zorder = zorder_value)
    # plt.fill_betweenx(mean_pred, mean_obs - std_obs, mean_obs + std_obs,
    #                  color=method_color, alpha=0.1)

    plt.fill_between(mean_obs, mean_pred - std_pred, mean_pred + std_pred,
                 color=colors[client_id][method_name], alpha=0.1, zorder = zorder_value)



        # Drop NaNs
    mask = ~np.isnan(df['pred_risk']) & ~np.isnan(df['event_at_t_star'])
    X = df.loc[mask, 'pred_risk'].values.reshape(-1, 1)
    y = df.loc[mask, 'event_at_t_star'].values

    cal_model = LogisticRegression().fit(X, y)
    slope = cal_model.coef_[0][0]
    intercept = cal_model.intercept_[0]
    print ("method_name", method_name)
    print(f"Calibration slope: {slope:.3f}, intercept: {intercept:.3f}")





# Labels and legend
plt.xlabel("Observed event rate", fontsize=14)
plt.ylabel("Mean predicted risk", fontsize=14)
# plt.title("Calibration Curves Across 10 Methods")

if dataset=="whas":

    plt.legend(fontsize=12, loc='lower right')

    # Perfect calibration reference
    plt.plot([0, 1.05], [0, 1.05], '--', color='gray', label='Perfect calibration')
    # plt.plot([0, 0.3], [0, 0.3], '--', color='gray', label='Perfect calibration')

    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)

else:

    plt.legend(fontsize=12, loc='lower right')

    # Perfect calibration reference
    plt.plot([0, 0.605], [0, 0.605], '--', color='gray', label='Perfect calibration')

    plt.ylim(0, 0.605)
    plt.xlim(0, 0.605)

    # # Perfect calibration reference
    # plt.plot([0, 0.3], [0, 0.3], '--', color='gray', label='Perfect calibration')

    # plt.ylim(0, 0.3)
    # plt.xlim(0, 0.3)


# plt.grid(True)
# Final layout
# plt.tight_layout()

plt.yticks(fontsize=11)
plt.xticks(fontsize=11)

plt.savefig(os.path.join(calib_roc_dir, "calibration_curves_%s_%s.png" %(client_id, dataset)), dpi=1000, bbox_inches='tight')
# plt.savefig(os.path.join(calib_plots_dir, "calibration_lines_%s.eps" %category), format='eps', bbox_inches='tight')  # no transparency

