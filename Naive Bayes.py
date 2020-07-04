import numpy as np
import matplotlib as plt 
import pandas as pd

dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#Splitting data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y,test_size=0.25,random_state=0) #20% will go into testing set

#Feature Scaling. Not really necessary in this, but improves the prediction
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #applying on all features
X_test = sc.transform(X_test)

#Implementing Naive Bayes. Non-linear Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

#Predicting for age =30 est salary=87000
print(classifier.predict(sc.transform([[30,87000]])))


#Predicting for test data
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(y_pred),1)),1))

#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test,y_pred)
print(cm)
accuracy_score(Y_test,y_pred)

