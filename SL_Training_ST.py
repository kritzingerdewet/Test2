# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:31:31 2023

@author: Kritzinger
"""

#Importing libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import safe_sqr

import pandas as pd 
from numpy import mean 
from numpy import std 
from numpy import absolute 
from sklearn.datasets import make_regression 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RepeatedKFold 
from sklearn.multioutput import RegressorChain 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.svm import LinearSVR 
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from datetime import datetime
import operator
import pickle


X = pd.read_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Input_CD.csv') 
y = pd.read_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Target_CD.csv')


X = X.loc[:, ~X.columns.str.contains('^Unnamed')] 
y = y.loc[:, ~y.columns.str.contains('^Unnamed')]

X = X.tail(13140)
y = y.tail(13140)



#Training
print(datetime.now()) 
model = RegressorChain(LGBMRegressor()).fit(X, y)
print(datetime.now()) 






# Serialize with Pickle
with open(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\LGBM_CD.pkl', 'wb') as pkl:
    pickle.dump(model, pkl)


# Now read it back and make a prediction
with open(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\LGBM_DG.pkl', 'rb') as pkl:
    pickle_preds = pickle.load(pkl).predict(n_periods=24, X=train[-24:].drop(columns=['Date', 'Dispatchable Generation']), alpha=0.05)






























