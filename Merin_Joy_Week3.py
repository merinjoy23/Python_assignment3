#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sqlite3
import pandas as pd


# In[3]:


conn=sqlite3.connect('/Users/Merin/Desktop/HAP880/Assignment3/testClaims_hu.db') # enter full path here
df=pd.read_sql('select * from highUtilizationPredictionV2wco',conn)


# In[4]:


df=df.join(pd.get_dummies(df.race))
df.head()


# In[5]:


cols=list(df.columns)
cols.remove('index')
cols.remove('race')
cols.remove('patient_id')
cols.remove('HighUtilizationY2')
cols.remove('claimCount')


# In[6]:


from sklearn.model_selection import train_test_split
tr, ts = train_test_split(df)


# In[7]:


from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize


# ### Random Forest

# In[8]:


space = [Integer(10, 60, name='n_estimators'),
         Categorical(['sqrt', 'log2'], name='max_features'),
         Categorical(['gini','entropy'], name='criterion')]


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
@use_named_args(space)
def objective(**params):
    rf = RandomForestClassifier(**params)
    return -np.mean(cross_val_score(rf, tr[cols], tr['HighUtilizationY2'], cv=5, n_jobs=1, scoring='roc_auc'))


# In[10]:


reg_gp = gp_minimize(objective, space, verbose=True)


# In[11]:


reg_gp


# In[12]:


from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import numpy as np


# In[13]:


rf = RandomForestClassifier()
rf


# ### Random Forest - Default parameters

# In[14]:


res_rf = []
rf = RandomForestClassifier()
scores = [10, 'auto','gini', cross_validate(rf, tr[cols], tr['HighUtilizationY2'], 
                                            scoring=['roc_auc','accuracy','precision','recall'], cv=10)]
res_rf.append(scores)
res_df_rf = pd.DataFrame(columns=["Estimators","max_features","criterion","score","score_val0","score_val1","score_val2",
                                  "score_val3",
                                  "score_val4","score_val5","score_val6","score_val7","score_val8","score_val9",])
cnt=0
for r in res_rf:
    for k,v in r[3].items():
        l = [r[0]]
        l.append(r[1])
        l.append(r[2])
        l.append(k)
        for i in v:
            l.append(i)
        res_df_rf.loc[cnt]=l  
        cnt = cnt + 1
res_df_rf['average'] = res_df_rf[["score_val0","score_val1","score_val2","score_val3","score_val4","score_val5","score_val6",
                                  "score_val7","score_val8","score_val9"]].mean(numeric_only=True, axis=1)
res_df_rf


# ### Random Forest - Optimized parameters

# In[15]:


res_rf = []
rf=RandomForestClassifier(n_estimators = 60, max_features = 'log2',criterion = 'entropy')
scores = [60, 'log2','entropy', cross_validate(rf, tr[cols], tr['HighUtilizationY2'], 
                                            scoring=['roc_auc','accuracy','precision','recall'], cv=10)]
res_rf.append(scores)
res_df_rf = pd.DataFrame(columns=["Estimators","max_features","criterion","score","score_val0","score_val1","score_val2",
                                  "score_val3",
                                  "score_val4","score_val5","score_val6","score_val7","score_val8","score_val9",])
cnt=0
for r in res_rf:
    for k,v in r[3].items():
        l = [r[0]]
        l.append(r[1])
        l.append(r[2])
        l.append(k)
        for i in v:
            l.append(i)
        res_df_rf.loc[cnt]=l  
        cnt = cnt + 1
res_df_rf['average'] = res_df_rf[["score_val0","score_val1","score_val2","score_val3","score_val4","score_val5","score_val6",
                                  "score_val7","score_val8","score_val9"]].mean(numeric_only=True, axis=1)
res_df_rf


# ### Logistic Regression

# In[16]:


space_lr = [Real(0.01,1.0, name='C'),
         Categorical(['liblinear','lbfgs'], name='solver'),
         Categorical([None, 'balanced'], name='class_weight')]


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
@use_named_args(space_lr)
def objective(**params):
    lr = LogisticRegression(**params)
    return -np.mean(cross_val_score(lr, tr[cols], tr['HighUtilizationY2'], cv=5, n_jobs=1, scoring='roc_auc'))


# In[18]:


reg_gp_lr = gp_minimize(objective, space_lr, verbose=True)


# In[19]:


reg_gp_lr


# In[20]:


lr = LogisticRegression()
lr


# ### Logistic Regression - Default Parameters

# In[25]:


res_lr = []
lr = LogisticRegression()
scores = [1.0, 'liblinear','None', cross_validate(lr, tr[cols], tr['HighUtilizationY2'], 
                                            scoring=['roc_auc','accuracy','precision','recall'], cv=10)]
res_lr.append(scores)
res_df_lr = pd.DataFrame(columns=["C","solver","class_weight","score","score_val0","score_val1","score_val2",
                                  "score_val3",
                                  "score_val4","score_val5","score_val6","score_val7","score_val8","score_val9",])
cnt=0
for r in res_lr:
    for k,v in r[3].items():
        l = [r[0]]
        l.append(r[1])
        l.append(r[2])
        l.append(k)
        for i in v:
            l.append(i)
        res_df_lr.loc[cnt]=l  
        cnt = cnt + 1
res_df_lr['average'] = res_df_lr[["score_val0","score_val1","score_val2","score_val3","score_val4","score_val5","score_val6",
                                  "score_val7","score_val8","score_val9"]].mean(numeric_only=True, axis=1)
res_df_lr


# ### Logistic Regression - Optimized Parameters

# In[30]:


res_lr = []
lr = LogisticRegression(C=0.2, solver='liblinear',class_weight='balanced')
scores = [0.2, 'liblinear','balanced', cross_validate(lr, tr[cols], tr['HighUtilizationY2'], 
                                            scoring=['roc_auc','accuracy','precision','recall'], cv=10)]
res_lr.append(scores)
res_df_lr = pd.DataFrame(columns=["C","solver","class_weight","score","score_val0","score_val1","score_val2",
                                  "score_val3",
                                  "score_val4","score_val5","score_val6","score_val7","score_val8","score_val9",])
cnt=0
for r in res_lr:
    for k,v in r[3].items():
        l = [r[0]]
        l.append(r[1])
        l.append(r[2])
        l.append(k)
        for i in v:
            l.append(i)
        res_df_lr.loc[cnt]=l  
        cnt = cnt + 1
res_df_lr['average'] = res_df_lr[["score_val0","score_val1","score_val2","score_val3","score_val4","score_val5","score_val6",
                                  "score_val7","score_val8","score_val9"]].mean(numeric_only=True, axis=1)
res_df_lr


# ### Gradient Boosting

# In[31]:


space_gb = [Integer(100, 200, name='n_estimators'),
         Categorical(['sqrt', 'log2'], name='max_features'),
         Categorical(['deviance', 'exponential'], name='loss')]


# In[32]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
@use_named_args(space_gb)
def objective(**params):
    gb = GradientBoostingClassifier(**params)
    return -np.mean(cross_val_score(gb, tr[cols], tr['HighUtilizationY2'], cv=5, n_jobs=1, scoring='roc_auc'))


# In[33]:


reg_gp_gb = gp_minimize(objective, space_gb, verbose=True)


# In[34]:


reg_gp_gb


# In[35]:


gb = GradientBoostingClassifier()
gb


# ### Gradient Boosting - Default Parameters

# In[36]:


res_gb = []
gb = GradientBoostingClassifier()
scores = [100, 'None','deviance', cross_validate(gb, tr[cols], tr['HighUtilizationY2'], 
                                            scoring=['roc_auc','accuracy','precision','recall'], cv=10)]
res_gb.append(scores)
res_df_gb = pd.DataFrame(columns=["Estimators","max_features","loss","score","score_val0","score_val1","score_val2",
                                  "score_val3",
                                  "score_val4","score_val5","score_val6","score_val7","score_val8","score_val9",])
cnt=0
for r in res_gb:
    for k,v in r[3].items():
        l = [r[0]]
        l.append(r[1])
        l.append(r[2])
        l.append(k)
        for i in v:
            l.append(i)
        res_df_gb.loc[cnt]=l  
        cnt = cnt + 1
res_df_gb['average'] = res_df_gb[["score_val0","score_val1","score_val2","score_val3","score_val4","score_val5","score_val6",
                                  "score_val7","score_val8","score_val9"]].mean(numeric_only=True, axis=1)
res_df_gb


# ### Gradient Boosting - Optimized Parameters

# In[37]:


res_gb = []
gb = GradientBoostingClassifier(n_estimators = 200 , max_features = 'sqrt',loss = 'exponential')
scores = [200, 'sqrt','exponential', cross_validate(gb, tr[cols], tr['HighUtilizationY2'], 
                                            scoring=['roc_auc','accuracy','precision','recall'], cv=10)]
res_gb.append(scores)
res_df_gb = pd.DataFrame(columns=["Estimators","max_features","loss","score","score_val0","score_val1","score_val2",
                                  "score_val3",
                                  "score_val4","score_val5","score_val6","score_val7","score_val8","score_val9",])
cnt=0
for r in res_gb:
    for k,v in r[3].items():
        l = [r[0]]
        l.append(r[1])
        l.append(r[2])
        l.append(k)
        for i in v:
            l.append(i)
        res_df_gb.loc[cnt]=l  
        cnt = cnt + 1
res_df_gb['average'] = res_df_gb[["score_val0","score_val1","score_val2","score_val3","score_val4","score_val5","score_val6",
                                  "score_val7","score_val8","score_val9"]].mean(numeric_only=True, axis=1)
res_df_gb


# In[ ]:




