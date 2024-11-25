from lifelines.datasets import load_regression_dataset
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np

def fit_and_score_features2(X):
   y=X[["E","label"]]
   X.drop(["label", "E"], axis=1, inplace=True)
   n_features = X.shape[1]
   scores = np.empty(n_features)
   m = CoxPHFitter()

   for j in range(n_features):
       Xj = X.iloc[:, j:j+1]
       Xj=pd.merge(Xj, y,  how='right', left_index=True, right_index=True)
       m.fit(Xj, duration_col="label", event_col="E", show_progress=True)
       a = m.print_summary()


   return scores,a
data_csv = pd.read_csv("bin/csv_data/t2-re_mask-1-test.csv")
scores ,a= fit_and_score_features2(data_csv)
# print(a.coef)
# cph.print_summary()
# print(a)
# print(data_csv.head())
# print(type(data_csv))
# print(data_csv["label"].value_counts())