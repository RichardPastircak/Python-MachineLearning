# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 18:26:00 2021

@author: Mr.Panda
"""

import numpy as np
import math as m
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC



x_public = np.load('X_public.npy',  allow_pickle=True)
y_public = np.load('Y_public.npy',  allow_pickle=True)
x_eval = np.load('X_eval.npy',  allow_pickle=True)

    
#SOLVE STRS      
enc = LabelEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')



for i in range (0,20):
    label_encoder = enc.fit(x_public[::,180])
    integer_classes = label_encoder.transform(label_encoder.classes_)
    x_public[::,180]= label_encoder.transform(x_public[::,180])
    x_eval[::,180]=label_encoder.transform(x_eval[::,180])


    aa= np.array(x_public[::,180]).reshape(-1, 1)
    one_hot_encoder = ohe.fit(aa)
    ohe_coded=ohe.transform(aa).toarray() 

    aa1= np.array(x_eval[::,180]).reshape(-1, 1)
    ohe_coded1=ohe.transform(aa1).toarray() 

    x_public=pd.DataFrame(x_public)
    x_public = x_public.drop(columns=180)  
    
    x_eval=pd.DataFrame(x_eval)
    x_eval = x_eval.drop(columns=180)

    for j in range (len(ohe_coded[0])):
        x_public["class 1" + str(j+1)] = ohe_coded[:,j]   
        
    for k in range (len(ohe_coded1[0])):
        x_eval["class 1" + str(k+1)] = ohe_coded1[:,k]       
        
    x_public = x_public.to_numpy()  
    x_eval = x_eval.to_numpy()
   

#CLEAR NANS
imputer = SimpleImputer(strategy='median')
x_public[:,:180] = imputer.fit_transform(x_public[:,:180])
x_eval[:,:180] = imputer.transform(x_eval[:,:180])

'''      
average = 0.0
isnans = 0

for j in range (0,180):
    #count all numbers and number of nans
    for i in range (0,600):
        if(not m.isnan(x_public[i][j])):
            average += x_public[i][j]
        else:
            isnans += 1
    
    #count average value of column and replace nans
    average = average / (600-isnans)
    for i in range (0,600):
        if(m.isnan(x_public[i][j])):
            x_public[i][j] = average
    
    #reset variables
    average = 0.0
    isnans = 0
'''


#Scaling
scaler = StandardScaler()
scaler.fit(x_public[:, :180])
x_public[:,:180] = scaler.transform(x_public[:, :180])   
x_eval[:,:180] = scaler.transform(x_eval[:,:180])


#SPLIT PUBLIC DATA
x_train, x_test, y_train, y_test = train_test_split(
        x_public, y_public, test_size=0.25, random_state=0)


#Testing time
svc = SVC()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
nusvc = NuSVC()

#SVC 0.97
'''
grid_search_params_svc = {
    'C': [1,1.5,2,2.5,3,],
    'kernel': ['linear', 'poly','rbf','sigmoid'],
    'degree': [1,2,3,4,],
    'gamma': ['scale','auto']
    }

gs= GridSearchCV(svc, grid_search_params_svc, cv=5, scoring='roc_auc')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)



svc = SVC(C = 1, degree = 2, kernel = 'poly')
svc.fit(x_train,y_train)


y_predict = svc.predict(x_test) #na zaklade toho predikuj na novych vzorkach

print ("Presnost klasifikacie ", roc_auc_score(y_predict, y_test))
'''

#DTC 63
'''
grid_search_params_dtc = {
    'criterion': ['gini','entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [4,5,6,7,8,9,10,11,12,13],
    'min_samples_split': [4,5,6,7,8,9,10,11,12,13],
}

gs= GridSearchCV(dtc, grid_search_params_dtc, cv=5, scoring='roc_auc')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)


dtc =  DecisionTreeClassifier(criterion='entropy', max_depth=11, min_samples_split=5, splitter='random')
dtc.fit(x_train,y_train)

y_predict = dtc.predict(x_test) #na zaklade toho predikuj na novych vzorkach

print ("Presnost klasifikacie ", roc_auc_score(y_predict, y_test))
'''

#RFC 55
'''
grid_search_params_rfc = {
    'n_estimators': [306,307,308,309,310,],
    'criterion': ['gini','entropy'],
    'max_depth': [8,9,10,11,12,13],
    'min_samples_split': [3,4,5,6,7,8,9,10,],
}

gs= GridSearchCV(rfc, grid_search_params_rfc, cv=5, scoring='roc_auc')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)



rfc = RandomForestClassifier(max_depth=13, min_samples_split=7, n_estimators=310)
rfc.fit(x_train,y_train)

y_predict = rfc.predict(x_test) #na zaklade toho predikuj na novych vzorkach

print ("Presnost klasifikacie ", roc_auc_score(y_predict, y_test))
'''

#nuSVC 0.96
'''
grid_search_params_nusvc = {
    'nu': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'kernel': ['linear','poly','rbf','sigmoid'],
    'degree': [1,2,3,4,5,6,7,8],
    'gamma': ['scale','auto']
}

gs= GridSearchCV(nusvc, grid_search_params_nusvc, cv=5, scoring='roc_auc')
gs.fit(x_train, y_train)
print(gs.best_estimator_)
print(gs.best_score_)



nusvc = NuSVC(nu=0.9, degree = 2, kernel = 'poly') 
nusvc.fit(x_train,y_train)
    
y_predict = nusvc.predict(x_test) 
print ("Presnost klasifikacie ", roc_auc_score(y_predict, y_test))
'''

svc = SVC(C = 1, degree = 2, kernel = 'poly')
svc.fit(x_public,y_public)


y_predikcia = svc.predict(x_eval) 
np.save("y_predikcia", y_predikcia)




