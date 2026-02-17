"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import os,sys
import pandas as pd
from typing import Any

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.client import AlgorithmClient

from .utils import *
from .networks import DeepSurv
from .networks import NegativeLogLikelihood
from .datasets import EventDataset

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.optim as optim

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@data(1)
@algorithm_client
def partial_risk_prediction(
    client: AlgorithmClient, df1: pd.DataFrame, predictor_cols, outcome_cols, dl_config, avged_params, update_iter, n_fold, fold_index
) -> Any:

    """ Decentral part of the algorithm """

    # print ("client.node.get()", client.node.get() )
    client_id = client.node.get()["id"]
    print ("update_iter", update_iter)
    print ("client_id", client_id)

    imputer = IterativeImputer(random_state=0, max_iter=5)
    lenfol_col = df1['LENFOL']
    fstat_col = df1['FSTAT']
    df_for_imputation = df1.drop(columns=['LENFOL', 'FSTAT'])
    print (df_for_imputation.columns)
    print ("len(df_for_imputation)",  len(df_for_imputation))
    imputer.fit(df_for_imputation)
    imputed_array = imputer.transform(df_for_imputation)
    df_imputed = pd.DataFrame(imputed_array, columns = df_for_imputation.columns)
    df_imputed['LENFOL'] = lenfol_col
    df_imputed['FSTAT'] = fstat_col


    info("Computing on client side")
    # train_df, val_df, test_df = tt_split(df_imputed)
    train_df, val_df, test_df = ci_split(df_imputed, n_fold = n_fold, fold_index = fold_index)
    # print (train_df)
    # print ("dl_config", dl_config)

    ## Vertical data split: X (feature), e (FSTAT), y(LENFOL)
    y_col = [outcome_cols[0]]
    e_col = [outcome_cols[1]]

    train_X, train_e, train_y = vertical_split(train_df, predictor_cols, y_col, e_col)
    val_X, val_e, val_y = vertical_split(val_df, predictor_cols, y_col, e_col)
    test_X, test_e, test_y = vertical_split(test_df, predictor_cols, y_col, e_col)

    train_X, X_min, X_max = normalize_train(train_X) # Normalize X
    val_X = normalize_test(val_X, X_min, X_max) # Nomralize val/test X based on min/max of train X
    test_X = normalize_test(test_X, X_min, X_max) # Nomralize val/test X based on min/max of train X

    # ## Convert numpy to torch tensor
    # train_X, train_e, train_y = np2tensor(train_X, train_e, train_y)
    # val_X, val_e, val_y = np2tensor(val_X, val_e, val_y)
    # test_X, test_e, test_y = np2tensor(test_X, test_e, test_y)

    train_dataset = EventDataset(train_X, train_e, train_y)
    val_dataset = EventDataset(val_X, val_e, val_y)
    test_dataset = EventDataset(test_X, test_e, test_y)


    ## save train e,y for time dependent AUC ROC
    # Create DataFrame
    train_y_array = [item[0] for item in train_y]
    train_e_array = [item[0] for item in train_e]

    df_train_output = pd.DataFrame({
        "y": train_y_array,
        "e": train_e_array
    })

    # Save df to the csv file  
    df_train_output.to_csv("df_train_followup_%s_%s.csv" %(client_id, fold_index), index=False)


    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    test_data_size = len(test_dataset)   

    # print ("train_X", train_X)

    print ("train_data_size", train_data_size)
    print ("val_data_size", val_data_size)
    print ("test_data_size", test_data_size)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=train_dataset.__len__())
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=val_dataset.__len__())

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=test_dataset.__len__())

    # batchsize = 256
    batchsize = 4096

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_dataset.__len__())

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())


    model = DeepSurv(dl_config['network']).to(device)
    # criterion = CrossEntropyLoss()

    learning_rate = dl_config['train']['learning_rate']
    if avged_params is not None:
        # Use global weight by fedavg
        avged_params = json2dict(avged_params)
        model.load_state_dict(avged_params)
        learning_rate = dl_config['train']['learning_rate']/10
        # learning_rate = dl_config['train']['learning_rate']

    criterion = NegativeLogLikelihood(dl_config['network'], device).to(device)

    optimizer = eval('optim.{}'.format(dl_config['train']['optimizer']))(
        model.parameters(), lr=learning_rate)



    train_loss_list = []
    val_loss_list = []
    train_ci_list = []
    val_ci_list = []

    for epoch in range(1, dl_config['train']['epochs']+1):
        # train step


        total_train_c = 0.0
        total_val_loss = 0
        total_train_loss = 0
        
        total_train_step = 0.0
        train_total_c = 0.0
        train_total_tp = 0
        val_total_tp = 0


        print ("epoch", epoch)
        model.train()
        for X, y, e in train_loader:
            # makes predictions
            # print ("device", device)
            X = X.to(device)
            y = y.to(device)
            e = e.to(device)
            
            risk_pred = model(X)
            risk_pred = risk_pred.to(device)

            train_loss = criterion(risk_pred, y, e, model)
            total_train_loss = total_train_loss + train_loss.item()
            train_c = c_index(-risk_pred, y, e)
            total_train_c = total_train_c + train_c
            # train_ci_list.append(train_c)
            total_train_step += 1.0
            # print ("train_c", train_c)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_loss_list.append(total_train_loss/total_train_step)
        # train_loss_list.append(total_train_loss)
        train_ci_list.append(total_train_c/total_train_step)



        model.eval()
        with torch.no_grad():
            for X, y, e in val_loader:

                X = X.to(device, dtype=float)
                y = y.to(device, dtype=float)
                e = e.to(device, dtype=float)

                # e = e.flatten()
                # e = e.type(torch.LongTensor) 
                # e = e.to(device)

                risk_pred = model(X)
                risk_pred = risk_pred.to(device)
                valid_loss = criterion(risk_pred, y, e, model)
                valid_c = c_index(-risk_pred, y, e)
                print ("valid_c", valid_c)
                val_ci_list.append(valid_c)
                val_loss_list.append(valid_loss.item())


    ## Plot training curves
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figure_temp_dir = os.path.join(current_dir, "figure_ci_%s" %fold_index )
    if not os.path.exists(figure_temp_dir):
        os.makedirs(figure_temp_dir)

    train_log={"train_c-index": train_ci_list, "val_c-index": val_ci_list, "train_loss": train_loss_list, "val_loss": val_loss_list}
    plot_train_curve_ci(train_log, figure_temp_dir, update_iter, client_id)

   ## Test trained model and save 
    test_cm_dict = {}
    test_eval_dict = {}

    output_list = []
    label_list = []
    pred_class_list =[]

    test_total_tp = 0


    model.eval()
    with torch.no_grad():
        for X, y, e in test_loader:

            X = X.to(device, dtype=float)
            y = y.to(device, dtype=float)
            e = e.to(device, dtype=float)
            print ("len(y)", len(y))
            # e = e.flatten()
            # e = e.type(torch.LongTensor) 
            # e = e.to(device)

            risk_pred = model(X)
            risk_pred = risk_pred.to(device)
            test_c = c_index(-risk_pred, y, e)
            print ("test_c", test_c)






    test_cm_dict['risk_pred'] = risk_pred.numpy().tolist()
    test_cm_dict['y'] = y.numpy().tolist()
    test_cm_dict['e'] = e.numpy().tolist()

    test_eval_dict['ci'] = test_c.tolist()



    ## return client's weights (after local training)
    ## https://github.com/itslastonenikhil/federated-learning/blob/main/FederatedLearning.ipynb
    model_params = model.state_dict()

    for entry in model_params:
        model_params[entry] = model_params[entry].cpu().data.numpy().tolist()


    model_params_json = json.dumps(model_params)
    

    test_cm_dict = json.dumps(test_cm_dict)
    test_eval_dict = json.dumps(test_eval_dict)



    # Return results to the vantage6 server.
    return {"params": model_params_json, "client_id":client_id, "num_train_samples": train_data_size, "test_cm": test_cm_dict, "test_eval": test_eval_dict}


# TODO Feel free to add more partial functions here.
