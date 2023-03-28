# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:53:20 2023

@author: Kritzinger
"""
import pandas as pd 
from pandas import Series 
from pandas import DataFrame 
from pandas import concat
import datetime as dt
import pickle

















#Load Model
pickled_model = pickle.load(open(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\LGBM_DG.pkl', 'rb'))



















pickled_model.predict(X_test)
    
    