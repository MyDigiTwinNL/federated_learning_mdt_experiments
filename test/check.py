import os,sys
import pandas as pd

filepath = "/groups/umcg-lifelines/tmp01/projects/ov22_0581/hmo/lifelines_preliminary/internal_data/datasets/cleaned_patients_df.csv"


df = pd.read_csv(filepath)
df = df.head(100)

print ("df", df)

