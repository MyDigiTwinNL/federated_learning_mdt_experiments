"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""
import os,sys
import json
import torch
import random
import numpy as np
import pandas as pd
from typing import Any
from .utils import *
from sklearn.metrics import confusion_matrix
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
random.seed(0)
np.random.seed(0)


@algorithm_client
def central_ci(
    client: AlgorithmClient, predictor_cols, outcome_cols, dl_config, num_update_iter, n_fold, fold_index
) -> Any:

    """ Central part of the algorithm """
    # TODO implement this function. Below is an example of a simple but typical
    # central function.

    # get all organizations (ids) within the collaboration so you can send a
    # task to them.
    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]

    global_ci_list = []
    local_ci_list = []

    ## Plot training curves
    current_dir = os.path.dirname(os.path.abspath(__file__))

    output_raw_dir = os.path.join(current_dir, "output_raw")
    if not os.path.exists(output_raw_dir):
        os.makedirs(output_raw_dir)

    output_raw_subdir_fold = os.path.join(output_raw_dir, str(fold_index))
    if not os.path.exists(output_raw_subdir_fold):
        os.makedirs(output_raw_subdir_fold)        

    ttest_dir = os.path.join(current_dir, "ttest_ci")
    if not os.path.exists(ttest_dir):
        os.makedirs(ttest_dir)

    for i in range(num_update_iter):
        print ("update iteration: ", i)

        if i == 0:
            avged_params = None
        else:
            avged_params = dict2json(avged_params)

        # Define input parameters for a subtask
        info("Defining input parameters")
        input_ = {
            "method": "partial_risk_prediction",
            "kwargs": {
                "predictor_cols": predictor_cols,
                "outcome_cols": outcome_cols,
                "dl_config": dl_config,
                "avged_params": avged_params,
                "update_iter": i,
                "n_fold": n_fold,
                "fold_index": fold_index

            }
        }


        # create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=org_ids,
            name="FedAvg_MDT",
            description="Training task on each client"
        )


        # wait for node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))

        info("Results obtained!")


        params_list =[]
        num_train_samples_list = []

        test_risk_pred_list = []
        test_y_list = []
        test_e_list = []

        test_ci_list = []

        # print ("results", results)

        for node_id, r in enumerate(results):
            params = r["params"]
            num_train_samples = r["num_train_samples"]
            params = json.loads(params) 

            client_id = r["client_id"]



            test_cm = r["test_cm"]
            # test_cm = json2dict(test_cm)
            test_cm = json.loads(test_cm) 

            test_eval = r["test_eval"]
            # test_eval = json2dict(test_eval)
            test_eval = json.loads(test_eval) 

            for entry in params:
                # params[entry] = torch.from_numpy(np.array(params[entry]))
                params[entry] = np.array(params[entry])


            test_risk_pred_list.append(test_cm['risk_pred'])
            test_y_list.append(test_cm['y'])
            test_e_list.append(test_cm['e'])
 
            test_ci_list.append(test_eval['ci'])


            params_list.append(params)
            num_train_samples_list.append(num_train_samples)


            ## Save outputs of local models (risk_pred, y, e)
            # Flatten the nested lists
            flat_pred_risk = [item[0] for item in test_cm['risk_pred']]
            flat_y = [item[0] for item in test_cm['y']]
            flat_e = [item[0] for item in test_cm['e']]

            # Create DataFrame
            df_raw_output = pd.DataFrame({
                "pred_risk": flat_pred_risk,
                "y": flat_y,
                "e": flat_e
            })

            # Save df to the csv file  
            temp_filepath = os.path.join(output_raw_subdir_fold, "df_output_raw_%s_%s.csv" %(client_id, i))
            df_raw_output.to_csv(temp_filepath, index=False)



        avged_params = fed_avg(params_list, num_train_samples_list)
        # print ("avged_params", avged_params)
        

        ## Compute global results
        risk_pred_stack = np.concatenate( test_risk_pred_list, axis=0 )
        y_stack = np.concatenate( test_y_list, axis=0 )
        e_stack = np.concatenate( test_e_list, axis=0 )



        ## aggregated globale results

        flat_pred_risk_glo = [item[0] for item in risk_pred_stack]
        flat_y_glo = [item[0] for item in y_stack]
        flat_e_glo = [item[0] for item in e_stack]

        # Create DataFrame
        df_raw_output_glo = pd.DataFrame({
            "pred_risk": flat_pred_risk_glo,
            "y": flat_y_glo,
            "e": flat_e_glo
        })

        # Save df to the csv file  
        temp_filepath_glo = os.path.join(output_raw_subdir_fold, "df_output_raw_glo_%s.csv" %i)
        df_raw_output_glo.to_csv(temp_filepath_glo, index=False)



        glo_ci = c_index(-risk_pred_stack, y_stack, e_stack)


        global_ci_list.append(glo_ci)

        local_ci_list.append(test_ci_list)


        figure_result_dir = os.path.join(current_dir, "figure_ci_results_%s" %fold_index)
        if not os.path.exists(figure_result_dir):
            os.makedirs(figure_result_dir)

        plot_global_results(global_ci_list, figure_result_dir, "C-statistic")


        plot_local_results(local_ci_list, figure_result_dir, "C-statistic")


        print ("global_ci_list", global_ci_list)
        print ("local_ci_list", local_ci_list)




    np.save(os.path.join(ttest_dir, "global_ci_%s.npy" %fold_index ), np.array(global_ci_list))
    np.save(os.path.join(ttest_dir, "local_ci_%s.npy" %fold_index), np.array(local_ci_list))



    return i




    # TODO probably you want to aggregate or combine these results here.
    # For instance:
    # results = [sum(result) for result in results]

    # return the final results of the algorithm


# TODO Feel free to add more central functions here.
