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
#import tensorflow_addons as tfa
#
# build an F1-score function for later use
#
def F1_score(y_true,y_prediction,true_class,true_threshold):
    T = len(y_true)
    if len(y_prediction) != T:
        print("Prediction and true label arrays have different size. Stop")
        return
    P = 0
    TP = 0 
    FN = 0
    TN = 0
    FP = 0
    for i in range(T):
        if y_true[i] == true_class:
            P = P + 1       
            if y_prediction[i] >= true_threshold:
                TP += 1 
            else:
                FN += 1
        else:
            if y_prediction[i] >= true_threshold:
                FP += 1 
            else:
                TN += 1            
    N = T - P    
    if TP == 0 and FP == 0 and FN == 0:
        F1 = 0
    else:
        F1 = 2.*TP/(2.*TP + FP + FN)
        Recall = TP/float(TP+FN)
    if TP == 0 and FP == 0: 
        Precision = 0.
    else:    
        Precision = TP/float(TP+FP)
    return F1, Recall, Precision
#
# function to read and filter data
#
def filterdata(infile,leadtime,varlist=[]):
    df = pd.read_csv(infile)  
    df.drop(['Storm'], axis=1, inplace=True)
    if leadtime == "24h":          
        for var in varlist:
            # axis = 0/1 means row/col drop 
            df.drop([var+'+00h',var+'+06h',var+'+12h',var+'+18h',var+'+24h'], axis=1, inplace=True)                 
    elif leadtime == "00h":
        fulllist = ['OHC','LAT','LON','MAXWIND','RMW','MIN_SLP','SHR_MAG','SHR_HDG',
                    'STM_SPD','STM_HDG','SST','TPW','LAND','850TANG','850VORT',
                    '200DVRG']
        for var in fulllist:
            df.drop([var+'+06h',var+'+12h',var+'+18h',var+'+24h'], axis=1, inplace=True)
        for var in varlist:
            df.drop([var+'+00h'], axis=1, inplace=True)
    else:
        print("flag_input_future_time is not correctly set. Stop")
        sys.exit()        
    df.replace('?',-99999, inplace=True)
    #print(df.head(5))
    return df
#
# visualization for logistics
#
def visualization_logistics(history_logistics):
    import matplotlib.pyplot as plt
    #print(history_logistics.__dict__)
    epochs = history_logistics.epoch
    val_loss = history_logistics.history['val_loss']
    loss = history_logistics.history['loss']
    plt.plot(epochs,val_loss,'r',label="Validation loss")
    plt.xlabel("epochs")
    plt.ylabel("value")
    plt.title("Loss history")
    plt.plot(epochs,loss,'b',label="Training loss")
    plt.legend()
    plt.show()
    
    plt.clf()
    val_accuracy = history_logistics.history['val_binary_accuracy']
    accuracy = history_logistics.history['binary_accuracy']
    plt.plot(epochs,val_accuracy,'r',label="Validation accuracy")
    plt.xlabel("epochs")
    plt.ylabel("value")
    plt.title("Accuracy history")
    plt.plot(epochs,accuracy,'b',label="Accuracy")
    plt.legend()
    plt.show()
#
# ploting the performance of the RNN model
#
def visualization_RNN(history_RNN):
    epoch = history_RNN.epoch
    val_loss_RNN = history_RNN.history['val_loss'] 
    loss_RNN = history_RNN.history['loss']
    plt.plot(epoch,loss_RNN,'r',label="loss_RNN")
    plt.xlabel("epoches")
    plt.ylabel("Value")
    plt.plot(epoch,val_loss_RNN,'b',label="val_loss_RNN")
    plt.legend()
    plt.show()
    
    accuracy_RNN = history_RNN.history['binary_accuracy']
    val_accuracy_RNN = history_RNN.history['val_binary_accuracy']
    plt.plot(epoch,accuracy_RNN,'r',label="RNN accuracy")
    plt.plot(epoch,val_accuracy_RNN,'b',label="RNN val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.show()    
#
# ploting the performance of the GRU model
#
def visualization_GRU(history_GRU):
    epoch = history_GRU.epoch
    val_loss_GRU = history_GRU.history['val_loss'] 
    loss_GRU = history_GRU.history['loss']
    plt.plot(epoch,loss_GRU,'r',label="loss_GRU")
    plt.xlabel("epoches")
    plt.ylabel("Value")
    plt.plot(epoch,val_loss_GRU,'b',label="val_loss_GRU")
    plt.legend()
    plt.show()
    
    accuracy_GRU = history_GRU.history['binary_accuracy']
    val_accuracy_GRU = history_GRU.history['val_binary_accuracy']
    plt.plot(epoch,accuracy_GRU,'r',label="GRU accuracy")
    plt.plot(epoch,val_accuracy_GRU,'b',label="GRU val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.show()
