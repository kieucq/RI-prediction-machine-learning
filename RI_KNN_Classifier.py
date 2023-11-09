import pandas as pd
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, svm, neighbors
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


#
# reading data
#
df = pd.read_csv('MajorAtlHurricaneMaster2016-2019.csv')
df.replace('?',-99999, inplace=True)
df.drop(['OHC'], 1, inplace=True)
print(df.head(10))
#
# creating (x,y) data from input data
#
x = np.array(df.drop(['class'],1))
y = np.array(df['class'])
#
# split data into 1 and 0 datasets
#

#
# create a new input data using the 1 data and a subsample of 0 data with
# the same length to balance the input data
#

#
# feed data into the KNN
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
pred = clf.predict(x_test)


cm = pd.DataFrame(confusion_matrix(y_test,pred))

print(accuracy)
print(classification_report(y_test,pred,zero_division=1))

classesNames = ["NoRi","RI"]
sns.heatmap(cm,annot=True,fmt="d",xticklabels=classesNames,yticklabels=classesNames)
plt.show()

#
# make a prediction with two sample inputs
#


"""
test_fsct = np.array([[4,2,1,1,3,2,4,1,1],[8,9,1,1,3,2,4,1,1]])
test_fsct = test_fsct.reshape(len(test_fsct),-1)
prediction = clf.predict(test_fsct)
print(prediction)
"""
