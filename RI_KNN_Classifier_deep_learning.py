import pandas as pd
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, svm, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('MajorAtlHurricaneMaster2016-2019.csv')
df.replace('?',-99999, inplace=True)
df.drop(['OHC'], axis=1, inplace=True)
print(df.head(10))
x = np.array(df.drop(['class'],axis=1)).astype("float32")
y = np.array(df['class']).astype("float32")
print(x[0],y[0])

from tensorflow import keras
from tensorflow.keras import layers

x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, test_size=0.2)
print(x1_train.shape,x1_test.shape,y1_train.shape,y1_test.shape)
x2_val = x1_train[:30]
y2_val = y1_train[:30]
x2_train = x1_train[30:]
y2_train = y1_train[30:]
print(y2_train)
print(x2_train.shape,x2_val.shape,y2_train.shape,y2_val.shape)

model = keras.Sequential([
    layers.Dense(16, activation = "relu"),
    layers.Dense(16, activation = "relu"),
    layers.Dense(1, activation = "sigmoid")])

model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])

history = model.fit(x2_train,y2_train,epochs=50,batch_size=32,validation_data=(x2_val,y2_val))

single_fcst = model.predict(x1_test)
for i in range(len(single_fcst)):
    print(y1_test[i]," <---> RI Propbability: ",single_fcst[i])

