#
# NOTE: This Stage-2 script test an RI prediction workflow based on 3 deep learning algorithms
#       that treat RI as yes/no (1/0) event. The workflow requires SHIP input predictors
#       from HWRF/HAFS forecasts from 0h to 24h lead time as an input.
#
#       To run in a job submission mode, use a Python script RI_deep_learning_master.py,
#       which is generated directly from this Jupyter notebook.
#
#       For more operational process, use 2 Python scripts RI_deep_learning_build.py
#       for training models, and RI_deep_learning_prediction.py for predicting step. These
#       two Python scripts should also be identical to this Jupyter notebook script,
#       except that the testing step is splitted and no visualization of the training can
#       be seen.
#
#       To add new DL models, edit the module file RI_deep_learning_libmodel.py
#
# HIST: - 10, Mar 2023: created by CK
#       - 02, Nov 2023: CK updated the input data for consistency of header information
#                       between training and real-time dataframes.
#       - 03, Nov 2023: CK re-worked on the flow to have both 00h and 24h predictors input
#
# AUTH: Chanh Kieu (ckieu@indiana.edu)
#===========================================================================================
import pandas as pd
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, svm, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import sys
import RI_deep_learning_libmodel as RImodel
import RI_deep_learning_libutils as RIutils
#
# set up initial parameters
#
debug = 0                       # debug printing level (0 or 1)
visualization = "no"            # displays the training history (yes or no). 
flag_input_future_time="24h"    # SHIP predictors input option (00h or 24h)
metric_threshold = 0.7          # binary accuracy threshold for training (0-1)
split_ratio = 0.05              # slit ratio between training/validation (0-1)
var_to_remove = ['OHC']         # list of vars to be removed. See the complete list above 
testfile="/N/u/ckieu/BigRed200/model/RI-prediction/OTIS18E_master.csv"
if flag_input_future_time == "24h":
    sequence_length = 5
elif flag_input_future_time == "12h":
    sequence_length = 3
elif flag_input_future_time == "00h":
    sequence_length = 1    
num_predictors = 16 - len(var_to_remove)
#
# reading a single case of a TC now for testing
#
df = RIutils.filterdata(testfile,flag_input_future_time,var_to_remove)
x_fcst = np.array(df.drop(['class'],axis=1))
y_true = np.array(df['class'])
x_tlag = x_fcst.reshape((-1,sequence_length,num_predictors))
if debug == 1:
    print('External input SHIP data length is: ',len(x_fcst))
    print('RI record for this storm is: ',y_true)
    print('Reshape for RNN and GRU input is: ',x_tlag.shape)
    print(x_fcst[0].astype('int'))
#
# Make prediction of RI for the single case (all cycles)
#
model_RNN = keras.models.load_model("RI_model_RNN_"+flag_input_future_time+".keras")
model_logistics = keras.models.load_model("RI_model_logistics_"+flag_input_future_time+".keras")
model_GRU = keras.models.load_model("RI_model_GRU_"+flag_input_future_time+".keras")

fcst_logistics = model_logistics.predict(x_fcst,verbose=debug)
fcst_GRU = model_GRU.predict(x_tlag,verbose=debug)
fcst_RNN = model_RNN.predict(x_tlag,verbose=debug)
for i in range(len(x_fcst)):
   print(f"Logistic, RNN, GRU probability predictions: {float(fcst_logistics[i]):.3f},{float(fcst_RNN[i]):.3f},{float(fcst_GRU[i]):.3f}")
print("F1, Recall, Precision for logistics model with "+flag_input_future_time+" data are:",RIutils.F1_score(y_true,fcst_logistics,1,0.1))
print("F1, Recall, Precision for RNN model with "+flag_input_future_time+" data are:",RIutils.F1_score(y_true,fcst_RNN,1,0.1))
print("F1, Recall, Precision for GRU model with "+flag_input_future_time+" data are:",RIutils.F1_score(y_true,fcst_GRU,1,0.1))  
