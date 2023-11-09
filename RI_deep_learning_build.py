#
# NOTE: This Stage-1 script builds an RI prediction workflow based on 3 deep learning algorithms
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
# reading all input data, and replace/remove bad data. Note that 
# - this data set must contain the last column "class" indicating RI (1) or non-RI (0). 
# - option flag_input_more_time must be 00h or 24h to indicate a single time slice, or 
#   mutiple time slices from SHIP forecasts.
# - A list of 16 vars to be removed must be consistent between models and prediction 
#   script. That is, var_to_remove must be set as one of the folliwng 17 variables
#   var_to_remove = ['OHC','LAT','LON','MAXWIND','RMW','MIN_SLP','SHR_MAG','SHR_HDG',
#                    'STM_SPD','STM_HDG','SST','TPW','LAND','850TANG','850VORT','200DVRG']                   
#
debug = 0                       # debug printing level (0 or 1)
visualization = "no"            # displays the training history (yes or no). 
flag_input_future_time="24h"    # SHIP predictors input option (00h or 24h)
metric_threshold = 0.7          # binary accuracy threshold for training (0-1)
split_ratio = 0.05              # slit ratio between training/validation (0-1)
var_to_remove = ['OHC']         # list of vars to be removed. See the complete list above 
infile='/N/u/ckieu/BigRed200/model/RI-prediction/SHIP_allbasin_2011_2022_Version4.csv'
df = RIutils.filterdata(infile,flag_input_future_time,var_to_remove)
if debug == 1: 
    print("The length of data frame df variable is ",len(df))
    print("The top 3 samples of the dataframe df are:")
    print(df.head(3))
#
# split (x,y) pair based on the lass col "class"
#
x = np.array(df.drop(['class'],axis=1)).astype("float32")
y = np.array(df['class']).astype("float32")
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, test_size=split_ratio,shuffle=True)
validation_split = int(split_ratio*len(x1_train))
x2_val = x1_train[:validation_split]
y2_val = y1_train[:validation_split]
x2_train = x1_train[validation_split:]
y2_train = y1_train[validation_split:]
if debug == 1:
    print(x[0],y[0])
    print("Total train and test data sizes are: ", x1_train.shape,x1_test.shape,y1_train.shape,y1_test.shape)
    print("Validation split is ",validation_split)
    print("Train and validation data sizes are: ", x2_train.shape,x2_val.shape,y2_train.shape,y2_val.shape)
    #print(x2_val[0].astype('int'),y2_val[0].astype('int'))
    #print(x1_train[385].astype('int'),y1_train[385].astype('int'))
    #print(x2_train[0].astype('int'),y2_train[0].astype('int'))
#
# Build a logistic model with options model_logistics32 or model_logistics64
#
model_logistics = RImodel.model_logistics32(metric_threshold=metric_threshold)
bestmodel_name = "RI_model_logistics_"+flag_input_future_time+".keras"
callbacks = [keras.callbacks.ModelCheckpoint(bestmodel_name,save_best_only=True)]
history_logistics = model_logistics.fit(x2_train,y2_train,epochs=100,batch_size=16,
                                       validation_data=(x2_val,y2_val),callbacks=callbacks,verbose=debug)
#
# check F1 score for the logistic model with the internal test data
#
if debug == 1:
    results = model_logistics.evaluate(x1_test,y1_test,verbose=debug)
    single_fcst = model_logistics.predict(x1_test,verbose=debug)
    #for i in range(len(single_fcst)):
    #    print(y1_test[i]," <---> RI Propbability: ",f"{float(single_fcst[i]):.5f}")
    print("Evalution results (loss,accuracy) for the test data is ",results)
    print("F1, Recall, Precision for logistic model are:",RIutils.F1_score(y1_test,single_fcst,1,0.10))  
#
# plotting the performance of the logistics regression
#
if visualization == "yes":
    RIutils.visualization_logistics(history_logistics)
#
# create input data for RNN by reshaping the input data into a new tensor of the dimension
# (num_sample, num_times, num_predictors). Note that using the metric=accuracy returns very bad
# accuracy < 0.1
# 
if flag_input_future_time == "24h":
    sequence_length = 5
elif flag_input_future_time == "12h":
    sequence_length = 3
elif flag_input_future_time == "00h":
    sequence_length = 1    
num_predictors = 16 - len(var_to_remove)
x3_val = x2_val.reshape((-1,sequence_length,num_predictors))
x3_train = x2_train.reshape((-1,sequence_length,num_predictors))
test_dataset = x1_test.reshape((-1,sequence_length,num_predictors))
if debug == 1:
    print("Train/val data sizes before reshape are: ", x2_train.shape,x2_val.shape,y2_train.shape,y2_val.shape)
    print("New train/validation data sizes for RNN are: ", x3_train.shape,x3_val.shape, test_dataset.shape)
    print(x2_val[0])
    print(x3_val[0])
#
# Build a RNN model with options model_RNN16 or model_RNN32
#
model_RNN = RImodel.model_RNN32(sequence_length,num_predictors,metric_threshold=metric_threshold)
bestmodel_name = "RI_model_RNN_"+flag_input_future_time+".keras"
callbacks = [keras.callbacks.ModelCheckpoint(bestmodel_name,save_best_only=True)]
history_RNN = model_RNN.fit(x3_train, y2_train, epochs=100, batch_size=64, 
                            validation_data=(x3_val, y2_val), callbacks=callbacks,verbose=debug)
#
# Check F1 score now for the RNN model
#
if debug == 1:
    model_best = keras.models.load_model(bestmodel_name)
    print(f"The best trained RNN prediction error is: {model_best.evaluate(test_dataset,y1_test,verbose=debug)[1]:.3f}")
    print(f"The last trained RNN prediction error is: {model_RNN.evaluate(test_dataset,y1_test,verbose=debug)[1]:.3f}")
    y_prediction = model_RNN.predict(test_dataset)
    print("F1, Recall, Precision for RNN model are:",RIutils.F1_score(y1_test,y_prediction,1,0.1))
#
# plotting the performance of the logistics regression
#
if visualization == "yes":
    RIutils.visualization_RNN(history_RNN)
#
# Build GRU model with options model_GRU16 or model_GRU32
#
model_GRU = RImodel.model_GRU32(sequence_length,num_predictors,metric_threshold=metric_threshold)
bestmodel_name = "RI_model_GRU_"+flag_input_future_time+".keras"
callbacks = [keras.callbacks.ModelCheckpoint(bestmodel_name,save_best_only=True)]
history_GRU = model_GRU.fit(x3_train, y2_train, epochs=100, batch_size=64, 
                             validation_data=(x3_val, y2_val), callbacks=callbacks,verbose=debug)
#
# Check F1 score now for the GRU model
#
if debug == 1:
    model_best = keras.models.load_model(bestmodel_name)
    print(f"The best trained GRU prediction error is: {model_best.evaluate(test_dataset,y1_test,verbose=debug)[1]:.3f}")
    print(f"The last trained GRU prediction error is: {model_GRU.evaluate(test_dataset,y1_test,verbose=debug)[1]:.3f}")
    y_prediction = model_GRU.predict(test_dataset)
    print("F1, Recall, Precision for GRU model are:",RIutils.F1_score(y1_test,y_prediction,1,0.1))
#
# plotting the performance of the logistics regression
#
if visualization == "yes":
    RIutils.visualization_GRU(history_GRU)

