
# coding: utf-8

# In[1]:

import csv
import numpy as np
import random as rand
from sklearn import linear_model, preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from scipy.io import loadmat, savemat


# In[2]:

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


# In[3]:

feat=loadmat("../../data/feat1.mat")


# In[4]:

#read training features
len_train=feat['len_train'][0][0]
x=np.concatenate((feat['x1_int'],feat['x1_float']),axis=1)
x=x[:len_train][:]

# In[5]:

#read training truth
y=feat['y'][:len_train]
y=y[:,0]


# In[ ]:

#read testing data
x_test=np.concatenate((feat['x1_int'],feat['x1_float']),axis=1)
x_test=x_test[len_train:][:]


# In[ ]:

best_c_abs=100
best_n_components_abs=122

best_c_01=100
best_n_components_01=125

#training(track 1)
pca = PCA(n_components=best_n_components_abs)
pca.fit(x)
x_reduce = pca.transform(x)
logistic = linear_model.LogisticRegression(C=best_c_abs)
logistic.fit(x_reduce,y)
#predict testing data
x_test_reduce=pca.transform(x_test)
y_test_predict = logistic.predict_proba(x_test_reduce)
y_test_predict = y_test_predict[:,1]
output_track1=np.vstack((x_test[:,0],y_test_predict))
output_track1=np.transpose(output_track1)


# In[ ]:

#write predict result to file
with open("../../result/PCA_logisticReg/temp_single_best/test_track1_v3.csv","w") as f:
    w=csv.writer(f)
    w.writerows(output_track1)
    f.close()


# In[ ]:

#training(track 2)
pca = PCA(n_components=best_n_components_01)
pca.fit(x)
x_reduce = pca.transform(x)
logistic = linear_model.LogisticRegression(C=best_c_01)
logistic.fit(x_reduce,y)
#predict testing data
x_test_reduce=pca.transform(x_test)
y_test_predict = logistic.predict(x_test_reduce)
output_track2=np.vstack((x_test[:,0],y_test_predict))
output_track2=np.transpose(output_track2)


# In[ ]:

with open("../../result/PCA_logisticReg/temp_single_best/test_track2_v3.csv","w") as f:
    w=csv.writer(f)
    w.writerows(output_track2)
    f.close()

