
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



def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """

    # Convert to floats, as python 2 rounds the divisions if we have integers
    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        print('[WORC Warning] Cannot compute a confidence interval for a single iteration.')
        print('[WORC Warning] CI will be set to value of single iteration.')
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

        metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

        CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    if np.isnan(CI[0]) and np.isnan(CI[1]):
        # When we cannot compute a CI, just give the averages
        CI = (metric_average, metric_average)
    return CI



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



update_iter_list = [20, 0]

model_results = {
    "0": {"0":{"valid_times": [], "auc_values": [], "mean_auc": [], "N_train": [], "N_test": []},"1":{"valid_times": [], "auc_values": [], "mean_auc": [], "N_train": [], "N_test": []},"2":{"valid_times": [], "auc_values": [], "mean_auc": [], "N_train": [], "N_test": []},"global":{"valid_times": [], "auc_values": [], "mean_auc": [], "N_train": [], "N_test": []}},
    "20": {"0":{"valid_times": [], "auc_values": [], "mean_auc": [], "N_train": [], "N_test": []},"1":{"valid_times": [], "auc_values": [], "mean_auc": [], "N_train": [], "N_test": []},"2":{"valid_times": [], "auc_values": [], "mean_auc": [], "N_train": [], "N_test": []},"global":{"valid_times": [], "auc_values": [], "mean_auc": [], "N_train": [], "N_test": []}}
}


for update_iter in update_iter_list:

    for fold_index in range(10):
        df_test_local_folder = os.path.join(output_raw_dir, str(fold_index))
        print ("fold_index", fold_index)
        # Load your CSV files
        list_train_followup_e = []
        list_train_followup_y = []
        for client_id in range(3):
            print ("client_id", client_id)

            df_train_filepath = os.path.join(output_raw_dir, "df_train_followup_%s_%s.csv" %(client_id, fold_index))
            df_train = pd.read_csv(df_train_filepath)
            
            df_test_local_filepath = os.path.join(df_test_local_folder, "df_output_raw_%s_%s.csv" %(client_id,update_iter))
            df_test = pd.read_csv(df_test_local_filepath)

            df_train = df_train[df_train["y"] < 3650.0]
            df_test = df_test[df_test["y"] < 3650.0]

            # Convert event columns to boolean
            e_train = df_train["e"].astype(bool).values
            y_train = df_train["y"].values/ 365.0

            num_train = len(y_train)

            list_train_followup_e.append(e_train)
            list_train_followup_y.append(y_train)

            train_structured = Surv.from_arrays(event=e_train, time=y_train)

            e_test = df_test["e"].astype(bool).values
            y_test = df_test["y"].values/ 365.0
            pred_risk = df_test["pred_risk"].values
            test_structured = Surv.from_arrays(event=e_test, time=y_test)

            num_test = len(y_test)
            # === STEP 1: Estimate censoring survival function (on train data) ===
            # We estimate it on train because that's what cumulative_dynamic_auc uses
            # censoring = (event == 0)
            censor_times, censor_survival = kaplan_meier_estimator(~e_train, y_train)

            # === STEP 2: Choose candidate times from test data
            candidate_times = np.percentile(y_test, np.linspace(10, 90, 9))

            # === STEP 3: Filter times where censoring survival is > 0
            valid_times = []
            for t in candidate_times:
                mask = censor_times >= t
                if np.any(mask):
                    s = censor_survival[mask][0]
                    if s > 0:
                        valid_times.append(t)

            valid_times = np.array(valid_times)
            # print ("valid_times", valid_times)

            if len(valid_times) < 2:
                raise ValueError("Too few valid evaluation times. Consider a different test/train split or larger sample.")


            # Compute time-dependent ROC AUC
            auc_values, mean_auc = cumulative_dynamic_auc(
                train_structured,
                test_structured,
                pred_risk,
                valid_times
            )

            # # Plot AUC vs Time
            # plt.figure(figsize=(8, 5))
            # plt.plot(valid_times, auc_values, marker='o')
            # plt.xlabel("Time (years)")
            # plt.ylabel("Time-dependent ROC AUC")
            # # plt.title("Time-dependent AUC over Time")
            # plt.ylim(0.0, 1.05)
            # plt.grid(True)
            # plt.tight_layout()
            # plt.savefig(os.path.join(calib_roc_dir, "time_dependent_auc_%s_%s_%s.png" %(update_iter, client_id, fold_index)), dpi=300, bbox_inches='tight')

            # # Print the integrated AUC
            # print(f"Integrated (mean) AUC: {mean_auc:.3f}")


            model_results[str(update_iter)][str(client_id)]["valid_times"].append(valid_times)
            model_results[str(update_iter)][str(client_id)]["auc_values"].append(auc_values)
            model_results[str(update_iter)][str(client_id)]["mean_auc"].append(mean_auc)
            model_results[str(update_iter)][str(client_id)]["N_train"].append(num_train)
            model_results[str(update_iter)][str(client_id)]["N_test"].append(num_test)


        ## Global    
        print ("Global")
        e_train = np.concatenate( list_train_followup_e, axis=0 )
        y_train = np.concatenate( list_train_followup_y, axis=0 )

        train_structured = Surv.from_arrays(event=e_train, time=y_train)

        df_test_local_filepath = os.path.join(df_test_local_folder, "df_output_raw_glo_%s.csv" %(update_iter))
        df_test = pd.read_csv(df_test_local_filepath)

        df_test = df_test[df_test["y"] < 3650.0]


        e_test = df_test["e"].astype(bool).values
        y_test = df_test["y"].values/ 365.0
        pred_risk = df_test["pred_risk"].values
        test_structured = Surv.from_arrays(event=e_test, time=y_test)

        num_train = len(y_train)
        num_test = len(y_test)

        # === STEP 1: Estimate censoring survival function (on train data) ===
        # We estimate it on train because that's what cumulative_dynamic_auc uses
        # censoring = (event == 0)
        censor_times, censor_survival = kaplan_meier_estimator(~e_train, y_train)

        # === STEP 2: Choose candidate times from test data
        candidate_times = np.percentile(y_test, np.linspace(10, 90, 9))

        # === STEP 3: Filter times where censoring survival is > 0
        valid_times = []
        for t in candidate_times:
            mask = censor_times >= t
            if np.any(mask):
                s = censor_survival[mask][0]
                if s > 0:
                    valid_times.append(t)

        valid_times = np.array(valid_times)
        # print ("valid_times", valid_times)

        if len(valid_times) < 2:
            raise ValueError("Too few valid evaluation times. Consider a different test/train split or larger sample.")

        # Compute time-dependent ROC AUC
        auc_values, mean_auc = cumulative_dynamic_auc(
            train_structured,
            test_structured,
            pred_risk,
            valid_times
        )

        # # Plot AUC vs Time
        # plt.figure(figsize=(8, 5))
        # plt.plot(valid_times, auc_values, marker='o')
        # plt.xlabel("Time")
        # plt.ylabel("Time-dependent ROC AUC")
        # plt.title("Time-dependent AUC over Time")
        # plt.ylim(0.0, 1.05)
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(calib_roc_dir, "time_dependent_auc_%s_glo_%s.png" %(update_iter, fold_index)), dpi=300, bbox_inches='tight')

        # # Print the integrated AUC
        # print(f"Integrated (mean) AUC: {mean_auc:.3f}")


        model_results[str(update_iter)]["global"]["valid_times"].append(valid_times)
        model_results[str(update_iter)]["global"]["auc_values"].append(auc_values)
        model_results[str(update_iter)]["global"]["mean_auc"].append(mean_auc)
        model_results[str(update_iter)]["global"]["N_train"].append(num_train)
        model_results[str(update_iter)]["global"]["N_test"].append(num_test)


print ("model_results", model_results)

client_lists=["0","1","2","global"]
for client_index in client_lists:
    for update_iter in update_iter_list:
        print ("update_iter", update_iter)
        print ("client_index", client_index)
        mean_auc_list_temp = model_results[str(update_iter)][client_index]["mean_auc"]
        n_train_list = model_results[str(update_iter)][client_index]["N_train"]
        n_test_list = model_results[str(update_iter)][client_index]["N_test"]
        n_train = sum(n_train_list) / len(n_train_list)
        n_test = sum(n_test_list) / len(n_test_list)
        folds_mean_auc = np.average(mean_auc_list_temp)
        CI = compute_confidence(mean_auc_list_temp, n_train, n_test, alpha=0.95)
        print ("folds_mean_auc", folds_mean_auc)
        print ("CI", CI)

        # compute mean and CI

model_name_list = ["FedAvg","Without aggregation"]

colors = {"0":{"Without aggregation": "purple", "FedAvg": "purple"}, "1":{"Without aggregation": "red", "FedAvg": "red"}, "2":{"Without aggregation": "blue", "FedAvg": "blue"}, "global":{"Without aggregation": "green", "FedAvg": "green"}}
linestyles = {"FedAvg": "-", "Without aggregation": "--"}

markers = {"FedAvg": "o", "Without aggregation": None}

client_index = "global"
plt.figure(figsize=(4, 3))
for update_iter, model_name in zip(update_iter_list, model_name_list):

    times_list = model_results[str(update_iter)][client_index]["valid_times"]
    aucs_list = model_results[str(update_iter)][client_index]["auc_values"]

    # Build common time grid
    common_times = np.linspace(
        min(min(t) for t in times_list),
        max(max(t) for t in times_list),
        num=50
    )

    # Interpolate each fold's AUC values
    interpolated_aucs = np.array([
        np.interp(common_times, t, auc)
        for t, auc in zip(times_list, aucs_list)
    ])

    # Mean and std AUC across folds
    mean_auc = interpolated_aucs.mean(axis=0)
    std_auc = interpolated_aucs.std(axis=0)

    # Plot mean line
    plt.plot(common_times, mean_auc, label=f"{model_name}", markevery=2,
            color=colors[client_index][model_name], linestyle=linestyles[model_name], marker = markers[model_name], linewidth=1.5)

    # # Optional: plot ±1 std as shaded band
    # plt.fill_between(common_times, mean_auc - std_auc, mean_auc + std_auc, 
    #                 color=colors[client_index][model_name], alpha=0.2)


# Final plot styling
plt.xlabel("Time (years)", fontsize=15)
plt.ylabel("Time-dependent AUC", fontsize=15)
# plt.title("Time-dependent AUC Comparison: Model A vs Model B")
# plt.ylim(0.65, 0.95)
plt.ylim(0.7, 0.9)
# plt.grid(True)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
# plt.legend(loc = "upper left", fontsize=10)
plt.legend( fontsize=11)
# plt.tight_layout()
plt.savefig(os.path.join(calib_roc_dir, "%s_time_dependent_auc_%s.png" %(dataset, client_index)), dpi=1000, bbox_inches='tight')

plt.close()

client_lists=["0","1","2"]
plt.figure(figsize=(4, 3))
for client_index in client_lists:
    model_name_local_list = ['FedAvg, client %s' %(int(client_index)+1), "Without aggr., client %s " %(int(client_index)+1)]
    for update_iter, model_name_local, model_name  in zip(update_iter_list, model_name_local_list, model_name_list):

        times_list = model_results[str(update_iter)][client_index]["valid_times"]
        aucs_list = model_results[str(update_iter)][client_index]["auc_values"]

        # Build common time grid
        common_times = np.linspace(
            min(min(t) for t in times_list),
            max(max(t) for t in times_list),
            num=50
        )

        # Interpolate each fold's AUC values
        interpolated_aucs = np.array([
            np.interp(common_times, t, auc)
            for t, auc in zip(times_list, aucs_list)
        ])

        # Mean and std AUC across folds
        mean_auc = interpolated_aucs.mean(axis=0)
        std_auc = interpolated_aucs.std(axis=0)

        # Plot mean line
        plt.plot(common_times, mean_auc, label=f"{model_name_local}", markevery=2,
                color=colors[client_index][model_name], linestyle=linestyles[model_name], marker = markers[model_name], linewidth=1.5, alpha=0.6)

        # # Optional: plot ±1 std as shaded band
        # plt.fill_between(common_times, mean_auc - std_auc, mean_auc + std_auc, 
        #                 color=colors[client_index][model_name], alpha=0.2)


# Final plot styling
plt.xlabel("Time (years)", fontsize=15)
plt.ylabel("Time-dependent AUC", fontsize=15)
# plt.title("Time-dependent AUC Comparison: Model A vs Model B")
# plt.ylim(0.65, 0.95)
plt.ylim(0.7, 0.9)
# plt.grid(True)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend(loc = "upper right", fontsize=8)
# plt.legend(loc = "upper left", fontsize=7)
# plt.tight_layout()
plt.savefig(os.path.join(calib_roc_dir, "%s_time_dependent_auc_local.png" %dataset), dpi=1000, bbox_inches='tight')

plt.close()