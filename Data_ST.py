# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:31:30 2023

@author: Kritzinger
"""


import pandas as pd 
from pandas import Series 
from pandas import DataFrame 
from pandas import concat
import datetime as dt

#Data Load & Format 

#df = pd.read_excel(r'C:\Users\Kritzinger\Documents\Eskom\ESK4296.xlsx') 
df = pd.read_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\StressTestSet.csv')
df2 = df

df2 = df2[['Date Time Hour Beginning', 'Original Res Forecast before Lockdown', 'Residual Forecast', 'RSA Contracted Forecast', 'Dispatchable Generation', 'Residual Demand', 'RSA Contracted Demand', 'International Exports', 'International Imports', 'Thermal Generation', 'Nuclear Generation', 'Eskom Gas Generation', 'Eskom OCGT Generation', 'Hydro Water Generation', 'Pumped Water Generation', 'ILS Usage', 'Manual Load_Reduction(MLR)', 'IOS Excl ILS and MLR', 'Dispatchable IPP OCGT', 'Eskom Gas SCO', 'Eskom OCGT SCO', 'Hydro Water SCO', 'Pumped Water SCO Pumping', 'Wind', 'PV', 'CSP', 'Other RE', 'Total RE', 'Wind Installed Capacity', 'PV Installed Capacity', 'CSP Installed Capacity', 'Other RE Installed Capacity', 'Total RE Installed Capacity', 'Installed Eskom Capacity', 'Total PCLF', 'Total UCLF', 'Total OCLF', 'Total UCLF+OCLF', 'Non Comm Sentout', 'Drakensberg Gen Unit Hours', 'Palmiet Gen Unit Hours', 'Ingula Gen Unit Hours'

     ]]
df2.rename(columns = {'Date Time Hour Beginning':'Date'}, inplace=True)

df3 = df2[['Date',
'Dispatchable Generation', 'Residual Demand', 'RSA Contracted Demand', 'International Exports', 'International Imports', 'Thermal Generation', 'Nuclear Generation', 'Eskom Gas Generation', 'Eskom OCGT Generation', 'Hydro Water Generation', 'Pumped Water Generation', 'ILS Usage', 'Manual Load_Reduction(MLR)', 'IOS Excl ILS and MLR', 'Dispatchable IPP OCGT', 'Eskom Gas SCO', 'Eskom OCGT SCO', 'Hydro Water SCO', 'Pumped Water SCO Pumping', 'Wind', 'PV', 'CSP', 'Other RE', 'Total RE', 'Wind Installed Capacity', 'PV Installed Capacity', 'CSP Installed Capacity', 'Other RE Installed Capacity', 'Total RE Installed Capacity', 'Installed Eskom Capacity', 'Total PCLF', 'Total UCLF', 'Total OCLF', 'Total UCLF+OCLF', 'Non Comm Sentout', 'Drakensberg Gen Unit Hours', 'Palmiet Gen Unit Hours', 'Ingula Gen Unit Hours'

     ]]

#Feature Engineering
####
df3['Date'] = pd.to_datetime(df3['Date'])
df3 = df3.dropna()

#Date/Time Features 
df3['Year'] = df3['Date'].dt.year 
df3['Month'] = df3['Date'].dt.month 
df3['day'] = df3['Date'].dt.day 
df3['hour'] = df3['Date'].dt.hour 
df3['DayOfWeek'] = df3['Date'].dt.dayofweek 
df3['WeekOfYear'] = df3['Date'].dt.weekofyear

days = {0:1,1:1,2:1,3:1,4:1,5:0,6:0}
df3['Week_Weekend'] = df3['DayOfWeek'].apply(lambda x: days[x])

hours = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0, #Non Business hours 
         8:1,9:1,10:1,11:1,12:1,13:1,14:1,15:1,16:1,#Business hours 
         17:2,18:2,19:2,20:2,#Evening Peak 
         21:0,22:0,23:0} #Non Business hours 
df3['BusinessHours'] = df3['hour'].apply(lambda x: hours[x])

months = {12:1,1:1,2:1, #Summer 
          3:2,4:2,5:2, #Autumn 
          6:3,7:3,8:3, #Winter 
          9:4,10:4,11:4} #Spring 
df3['Season'] = df3['Month'].apply(lambda x: months[x])


#Date limit
StressTestSet = df3[df3['Date'] >= '18-02-23 0:00'] 
StressTestSet.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\StressTestSet.csv')

df3 = df3[df3['Date'] < '18-02-23 0:00'] 

#Time series Structure
ts = df3.set_index('Date') 
ts = ts.dropna() 
ts.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\TimeSeriesData.csv')

#Supervised Structure 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True): 
    """ Frame a time series as a supervised learning dataset. Arguments: data: Sequence of observations as a list or NumPy array. n_in: Number of lag observations as input (X). n_out: Number of observations as output (y). dropnan: Boolean whether or not to drop rows with NaN values. Returns: Pandas DataFrame of series framed for supervised learning. """ 
    n_vars = 1 if type(data) is list  else data.shape[1] 
    df = DataFrame(data)
    cols, names = list(), list()

    #input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1): 
        cols.append(df.shift(i)) 
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    #forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out): 
        cols.append(df.shift(-i)) 
        if i == 0: 
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)] 
        else: 
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

     #  put it all together
    agg = concat(cols, axis=1) 
    agg.columns = names
 
#   drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True) 
    return agg



#Mapping 
columnNames = ts.columns 
columns = pd.DataFrame(columnNames).reset_index()


columns.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\ColumnNames.csv') 
values = ts.values 
values = values.astype('float32')

#frame as supervised learning
sl = series_to_supervised(values, 48, 168)

sl_2 = sl#sl.tail(20000) 
sl_2_columns = sl_2.columns 
sl_2_columns = pd.DataFrame(sl_2_columns).reset_index() 
sl_2_columns.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\sl2_columns.csv')


#Dispatchable Generation Split 
lag = sl_2.iloc[:,0:2256]
target = sl_2.iloc[:,[2256,	2303,	2350,	2397,	2444,	2491,	2538,	2585,	2632,	2679,	2726,	2773,	2820,	2867,	2914,	2961,	3008,	3055,	3102,	3149,	3196,	3243,	3290,	3337,	3384,	3431,	3478,	3525,	3572,	3619,	3666,	3713,	3760,	3807,	3854,	3901,	3948,	3995,	4042,	4089,	4136,	4183,	4230,	4277,	4324,	4371,	4418,	4465,	4512,	4559,	4606,	4653,	4700,	4747,	4794,	4841,	4888,	4935,	4982,	5029,	5076,	5123,	5170,	5217,	5264,	5311,	5358,	5405,	5452,	5499,	5546,	5593,	5640,	5687,	5734,	5781,	5828,	5875,	5922,	5969,	6016,	6063,	6110,	6157,	6204,	6251,	6298,	6345,	6392,	6439,	6486,	6533,	6580,	6627,	6674,	6721,	6768,	6815,	6862,	6909,	6956,	7003,	7050,	7097,	7144,	7191,	7238,	7285,	7332,	7379,	7426,	7473,	7520,	7567,	7614,	7661,	7708,	7755,	7802,	7849,	7896,	7943,	7990,	8037,	8084,	8131,	8178,	8225,	8272,	8319,	8366,	8413,	8460,	8507,	8554,	8601,	8648,	8695,	8742,	8789,	8836,	8883,	8930,	8977,	9024,	9071,	9118,	9165,	9212,	9259,	9306,	9353,	9400,	9447,	9494,	9541,	9588,	9635,	9682,	9729,	9776,	9823,	9870,	9917,	9964,	10011,	10058,	10105]] 
sl_2_DG = lag.join(target)


lag.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Input_DG.csv')
target.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Target_DG.csv')
sl_2_DG.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Full_DG.csv')






#Total RE
lag = sl_2.iloc[:,0:2279]
target = sl_2.iloc[:,[2279,	2326,	2373,	2420,	2467,	2514,	2561,	2608,	2655,	2702,	2749,	2796,	2843,	2890,	2937,	2984,	3031,	3078,	3125,	3172,	3219,	3266,	3313,	3360,	3407,	3454,	3501,	3548,	3595,	3642,	3689,	3736,	3783,	3830,	3877,	3924,	3971,	4018,	4065,	4112,	4159,	4206,	4253,	4300,	4347,	4394,	4441,	4488,	4535,	4582,	4629,	4676,	4723,	4770,	4817,	4864,	4911,	4958,	5005,	5052,	5099,	5146,	5193,	5240,	5287,	5334,	5381,	5428,	5475,	5522,	5569,	5616,	5663,	5710,	5757,	5804,	5851,	5898,	5945,	5992,	6039,	6086,	6133,	6180,	6227,	6274,	6321,	6368,	6415,	6462,	6509,	6556,	6603,	6650,	6697,	6744,	6791,	6838,	6885,	6932,	6979,	7026,	7073,	7120,	7167,	7214,	7261,	7308,	7355,	7402,	7449,	7496,	7543,	7590,	7637,	7684,	7731,	7778,	7825,	7872,	7919,	7966,	8013,	8060,	8107,	8154,	8201,	8248,	8295,	8342,	8389,	8436,	8483,	8530,	8577,	8624,	8671,	8718,	8765,	8812,	8859,	8906,	8953,	9000,	9047,	9094,	9141,	9188,	9235,	9282,	9329,	9376,	9423,	9470,	9517,	9564,	9611,	9658,	9705,	9752,	9799,	9846,	9893,	9940,	9987,	10034,	10081,	10128]] 
sl_2_DG = lag.join(target)


lag.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Input_RE.csv')
target.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Target_RE.csv')
sl_2_DG.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Full_RE.csv')





# CD
lag = sl_2.iloc[:,0:2258]
target = sl_2.iloc[:,[2258,	2305,	2352,	2399,	2446,	2493,	2540,	2587,	2634,	2681,	2728,	2775,	2822,	2869,	2916,	2963,	3010,	3057,	3104,	3151,	3198,	3245,	3292,	3339,	3386,	3433,	3480,	3527,	3574,	3621,	3668,	3715,	3762,	3809,	3856,	3903,	3950,	3997,	4044,	4091,	4138,	4185,	4232,	4279,	4326,	4373,	4420,	4467,	4514,	4561,	4608,	4655,	4702,	4749,	4796,	4843,	4890,	4937,	4984,	5031,	5078,	5125,	5172,	5219,	5266,	5313,	5360,	5407,	5454,	5501,	5548,	5595,	5642,	5689,	5736,	5783,	5830,	5877,	5924,	5971,	6018,	6065,	6112,	6159,	6206,	6253,	6300,	6347,	6394,	6441,	6488,	6535,	6582,	6629,	6676,	6723,	6770,	6817,	6864,	6911,	6958,	7005,	7052,	7099,	7146,	7193,	7240,	7287,	7334,	7381,	7428,	7475,	7522,	7569,	7616,	7663,	7710,	7757,	7804,	7851,	7898,	7945,	7992,	8039,	8086,	8133,	8180,	8227,	8274,	8321,	8368,	8415,	8462,	8509,	8556,	8603,	8650,	8697,	8744,	8791,	8838,	8885,	8932,	8979,	9026,	9073,	9120,	9167,	9214,	9261,	9308,	9355,	9402,	9449,	9496,	9543,	9590,	9637,	9684,	9731,	9778,	9825,	9872,	9919,	9966,	10013,	10060,	10107,]] 
sl_2_DG = lag.join(target)



lag.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Input_CD.csv')
target.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Target_CD.csv')
sl_2_DG.to_csv(r'C:\Users\Kritzinger\Documents\Eskom\Modelling\ModelStressTest\Full_CD.csv')









