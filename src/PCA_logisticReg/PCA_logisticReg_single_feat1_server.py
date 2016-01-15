
# coding: utf-8

# In[ ]:

import sys
import numpy as np
import random as rand
from sklearn import linear_model, preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from scipy.io import loadmat, savemat
import time


# In[ ]:

def E_abs(y_predict,y):
    E=0
    for i in range(len(y)):
        E+=abs(y_predict[i]-y[i])
    return E/len(y)

def E_01(y_predict,y):
    y_temp=[]
    for i in range(len(y)):
        if y_predict[i]>0.5:
            y_temp.append(1)
        elif y_predict[i]==0.5:
            y_temp.append(rand.randint(0,1))
        else:
            y_temp.append(0)
    return E_abs(y_temp,y)


# In[ ]:

feat=loadmat('../../data/feat1.mat')


# In[ ]:

#read training features
len_train=feat['len_train'][0][0]
x=np.concatenate((feat['x1_int'],feat['x1_float']),axis=1)
x=x[:len_train][:]
dimensions=len(x[0])


# In[ ]:

#read training truth
y=feat['y'][:len_train]
y=y[:,0]


# In[ ]:

#preprocessing the training data
# preprocessing.normalize(x, norm='l2')
# min_max_scaler = preprocessing.MinMaxScaler()
# x=min_max_scaler.fit_transform(x)


# In[ ]:

#training
n_folds=10
c_list=list(range(-4,5))
c_list[:]=[10 ** x for x in c_list]
n_components_list=list(range(int(dimensions/2),dimensions))
rand.shuffle(n_components)
best_Eval_01=1
best_Eval_abs=1
for c in c_list:
    for n_components in n_components_list:
        Ein_01=[]
        Ein_abs=[]
        Eval_01=[]
        Eval_abs=[]
        kf=KFold(len(x),n_folds=n_folds)
        for train_index, val_index in kf:
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]
            pca = PCA(n_components=n_components)
            pca.fit(x_train)
            x_train_reduce = pca.transform(x_train)
            logistic = linear_model.LogisticRegression(C=c)
            logistic.fit(x_train_reduce,y_train)
            y_train_predict = logistic.predict_proba(x_train_reduce)
            y_train_predict = y_train_predict[:,1]
            Ein_01.append(E_01(y_train_predict,y_train))
            Ein_abs.append(E_abs(y_train_predict,y_train))
            x_val_reduce = pca.transform(x_val)
            y_val_predict = logistic.predict_proba(x_val_reduce)
            y_val_predict = y_val_predict[:,1]
            Eval_01.append(E_01(y_val_predict,y_val))
            Eval_abs.append(E_abs(y_val_predict,y_val))
        if sum(Eval_01)/n_folds<best_Eval_01:
            best_Eval_01=sum(Eval_01)/n_folds
            best_n_components_01=n_components
            best_c_01=c
        if sum(Eval_abs)/n_folds<best_Eval_abs:
            best_Eval_abs=sum(Eval_01)/n_folds
            best_n_components_abs=n_components
            best_c_abs=c
        print("best_Eval_01:",best_Eval_01,"best_n_components",best_n_components_01,"best_c:",best_c_01)
        print("best_Eval_abs:",best_Eval_abs,"best_n_components",best_n_components_abs,"best_c:",best_c_abs)
        sys.stdout.flush()


# In[ ]:



