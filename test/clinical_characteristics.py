
import os, sys
import pandas as pd
import numpy as np


random_state = 0

def median_series(npseries):

    percentile = np.percentile(npseries, [25, 50, 75])

    return percentile


def mstd(npseries):

    return npseries.mean(), npseries.std()


def percent(npseries, totalnumber):


    return len(npseries)/totalnumber*100

def missingrate(npseries):

    return npseries.isnull().sum() * 100 / len(npseries)


def continueous(npseries):

    print ("mstd", mstd(npseries.dropna()))
    print ("median", median_series(npseries.dropna()))
    print ("missingrate", missingrate(npseries))

    return 


def discrete(npseries):

    print ("mstd", median_series(npseries.dropna()))
    # print ("median", median_series(npseries.dropna()))


    print ("missingrate", missingrate(npseries))



    return 

def main():

    ## Load CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(current_dir, "csv")
    data_filepath = os.path.join(csv_dir, "patient_t2e_df.csv")
    ## Load csv file before horizontal partitioning
    df_full = pd.read_csv(data_filepath)
    print (df_full.columns)
    # ['GENDER', 'T2D_STATUS', 'SMOKING_STATUS', 'SMOKING_QUANTITY',
    #    'TOTAL_CHOLESTEROL_VALUE', 'HDL_CHOLESTEROL_VALUE',
    #    'LDL_CHOLESTEROL_VALUE', 'SYSTOLIC_VALUE', 'DIASTOLIC_VALUE',
    #    'PLASMA_ALBUNIM_VALUE', 'EGFR_VALUE', 'HEMOGLOBIN_VALUE',
    #    'HYPERTENSION_STATUS', 'HBA1C_VALUE', 'CREATININE_VALUE', 'AGE',
    #    'LENFOL', 'FSTAT'],

    # # age, sex, eGFR, albumin, HDL cholesterol, LDL cholesterol, 
    # total cholesterol, HbA1c, hypertension history, type 2 diabetes history, 
    # creatinine, systolic blood pressure, diastolic blood pressure, smoking history, 
    # smoking quantity


    # # , , , , ,
    # , hypertension history, type 2 diabetes history, 
    # , , , 
    #

    ## Total number of ind
    print ("Total number of ind", len(df_full))

    ## Follow up time
    followup = median_series(df_full["LENFOL"])   
    print ("followup", followup/365.0)  

    ## Function for Mean(SD)
    df_women = df_full[df_full["GENDER"]==1]
    df_women_percent = percent(df_women, len(df_full))
    print ("len(df_women)", len(df_women), df_women_percent)

    ## Age
    print ("AGE")
    continueous(df_full["AGE"])

    ## 
    print ("SYSTOLIC_VALUE")
    continueous(df_full["SYSTOLIC_VALUE"])
    print ("DIASTOLIC_VALUE")
    continueous(df_full["DIASTOLIC_VALUE"])

    print ("HDL_CHOLESTEROL_VALUE")
    continueous(df_full["HDL_CHOLESTEROL_VALUE"])
    print ("LDL_CHOLESTEROL_VALUE")
    continueous(df_full["LDL_CHOLESTEROL_VALUE"])
    print ("TOTAL_CHOLESTEROL_VALUE")
    continueous(df_full["TOTAL_CHOLESTEROL_VALUE"])


    print ("EGFR_VALUE")
    continueous(df_full["EGFR_VALUE"])
    print ("PLASMA_ALBUNIM_VALUE")
    continueous(df_full["PLASMA_ALBUNIM_VALUE"])
    print ("HBA1C_VALUE")
    continueous(df_full["HBA1C_VALUE"])
    print ("CREATININE_VALUE")
    continueous(df_full["CREATININE_VALUE"])
    print ("SMOKING_QUANTITY")
    continueous(df_full["SMOKING_QUANTITY"])

    ## Smoking, Never smoked tobacco (finding) : 0, Ex-smoker (finding): 1, Smokes tobacco daily (finding): 2
    print ("SMOKING_STATUS")
    # discrete(df_full["SMOKING_STATUS"])
    print ("missingrate", missingrate(df_full["SMOKING_STATUS"]))
    df_t2 = df_full[df_full["SMOKING_STATUS"]==0]
    print ("len(s0)", len(df_t2))
    print ("percent", percent(df_t2, len(df_full["SMOKING_STATUS"].dropna())))
    df_t2 = df_full[df_full["SMOKING_STATUS"]==1]
    print ("len(s1)", len(df_t2))
    print ("percent", percent(df_t2, len(df_full["SMOKING_STATUS"].dropna())))
    df_t2 = df_full[df_full["SMOKING_STATUS"]==2]
    print ("len(s2)", len(df_t2))
    print ("percent", percent(df_t2, len(df_full["SMOKING_STATUS"].dropna())))


    ## T2D Status 'Active': 1.0 Nan: 0.0
    print ("T2D_STATUS")
    # discrete(df_full["T2D_STATUS"])

    print ("missingrate", missingrate(df_full["T2D_STATUS"]))
    df_t2 = df_full[df_full["T2D_STATUS"]==1]
    print ("len(df_t2)", len(df_t2))
    print ("percent", percent(df_t2, len(df_full["T2D_STATUS"].dropna())))


    print ("HYPERTENSION_STATUS")
    # discrete(df_full["HYPERTENSION_STATUS"])
    print ("missingrate", missingrate(df_full["HYPERTENSION_STATUS"]))
    df_t2 = df_full[df_full["HYPERTENSION_STATUS"]==1]
    print ("len(hstatus)", len(df_t2))
    print ("percent", percent(df_t2, len(df_full["HYPERTENSION_STATUS"].dropna())))

if __name__ == '__main__':
    main()    