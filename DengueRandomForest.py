# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:22:08 2018

@author: Adharsh
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

train = pd.read_csv("./data/dengue_features_train.csv")
train_labels = pd.read_csv("./data/dengue_labels_train.csv")
test = pd.read_csv("./data/dengue_features_test.csv")

train_target = train_labels['total_cases']

test_city = test['city']
test_year = test['year']
test_weekofyear = test['weekofyear']

df = train.append(test, ignore_index = True)
cats = []
for col in df.columns.values:
    if df[col].dtype == 'object':
        cats.append(col)
df_cat = df[cats]
df_cont = df.drop(cats, axis=1)
for col in df_cont.columns.values:
    #median values subsitituted for missing values
    df_cont[col] = df_cont[col].fillna(df[col].median())

#the week_start_date is not necessary as the week and year variables are already available
df_cat = df_cat.drop(['week_start_date'], axis = 1)

df_cat = pd.get_dummies(df_cat)

df = df_cont.join(df_cat)

train = df.iloc[0:train.shape[0]]
test = df.iloc[train.shape[0]:]
    
scorer = make_scorer(mean_absolute_error, False)
    
randForestModel = RandomForestRegressor(n_estimators=100, random_state=8)
crossValMean = np.sqrt(-cross_val_score(estimator=randForestModel, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).mean()
crossValSTD = np.sqrt(-cross_val_score(estimator=randForestModel, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).std()

randForestModel.fit(train, np.ravel(train_target))

predictions = randForestModel.predict(test)

predictions = predictions.astype(int)

submission = pd.DataFrame(predictions, columns=["total_cases"])

submission.insert(0, 'city', test_city)
submission.insert(1, 'year', test_year)
submission.insert(2, 'weekofyear', test_weekofyear)
submission.reset_index()
submission.to_csv('Submission.csv', index = False)