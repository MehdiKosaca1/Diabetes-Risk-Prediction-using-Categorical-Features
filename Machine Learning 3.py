#!/usr/bin/env python
# coding: utf-8
Data and classificaiton processes used
# In[100]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")
y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)

all libraries used
# In[98]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[99]:


import warnings
warnings.filterwarnings("ignore", category= DeprecationWarning)
warnings.filterwarnings("ignore", category= FutureWarning)


# # Logistic Regression
libraries used in this field
# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[4]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")


# In[5]:


data.head()


# In[6]:


data["Outcome"].value_counts()


# In[7]:


data.describe().T


# In[8]:


y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)


# In[9]:


y.head()


# In[10]:


x.head()


# In[121]:


loj_model = LogisticRegression(solver="liblinear").fit(x,y)


# In[13]:


loj_model.intercept_


# In[15]:


loj_model.coef_


# In[16]:


loj_model.predict(x)[0:10]


# In[17]:


y[0:10]


# In[18]:


y_pred= loj_model.predict(x)


# In[19]:


confusion_matrix(y,y_pred) #karmaşıklık matriksi


# In[20]:


accuracy_score(y,y_pred) #doğruluk oranı


# In[33]:


print(classification_report(y,y_pred))


# In[28]:


loj_model.predict_proba(x)[0:10]


# In[30]:


logit_roc_auc = roc_auc_score(y, loj_model.predict(x))

fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(x) [:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area 1 %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()


# In[34]:


x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)


# In[35]:


loj_model= LogisticRegression(solver= "liblinear").fit(x_train,y_train)


# In[36]:


y_pred= loj_model.predict(x_test)


# In[37]:


print(accuracy_score(y_test,y_pred))


# In[41]:


cross_val_score(loj_model, x_test, y_test, cv= 10).mean()


# # K-EN Yakın Komşu

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20183200.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20183200.png)
libraries used in this field
# In[3]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[43]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")


# In[44]:


y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)


# In[45]:


knn_model= KNeighborsClassifier().fit(x_train,y_train)


# In[46]:


knn_model


# In[47]:


y_pred= knn_model.predict(x_test)


# In[49]:


accuracy_score(y_test,y_pred)


# In[50]:


print(classification_report(y_test,y_pred))


# In[51]:


knn= KNeighborsClassifier()


# In[52]:


knn_params= {"n_neighbors":np.arange(1,50)}


# In[53]:


knn_cv_model= GridSearchCV(knn,knn_params, cv=10).fit(x_train,y_train)


# In[54]:


knn_cv_model.best_score_


# In[55]:


knn_cv_model.best_params_


# In[122]:


knn_tuned = KNeighborsClassifier(n_neighbors= 11).fit(x_train,y_train)


# In[60]:


y_pred= knn_tuned.predict(x_test)


# In[61]:


accuracy_score(y_test, y_pred)


# In[62]:


knn_tuned.score(x_test,y_test) # kısa yol


# # Support Vector Machines (SVM)
Purpose: to separate the two classes we have
# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20173930.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20173930.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20174035.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20174035.png)
libraries used in this field
# In[1]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[4]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")


# In[8]:


y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)


# In[11]:


svm_model= SVC(kernel= "linear").fit(x_train,y_train)


# In[12]:


svm_model


# In[13]:


y_pred= svm_model.predict(x_test)


# In[14]:


accuracy_score(y_test,y_pred)


# In[18]:


svm_params= {"C":np.arange(1,5),
            "kernel":["linear","rbf"]}


# In[17]:


svm= SVC()


# In[19]:


svm_cv_model= GridSearchCV(svm,svm_params,cv=5,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[20]:


svm_cv_model.best_score_


# In[21]:


svm_cv_model.best_params_


# In[123]:


svm_tuned= SVC(C= 2, kernel= "linear").fit(x_train,y_train)


# In[26]:


y_pred= svm_tuned.predict(x_test)


# In[27]:


accuracy_score(y_test,y_pred)


# # Artificial neural networks

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20190706.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20190706.png)
libraries used in this field
# In[66]:


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[28]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")
y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)

yapay sinir ağları genelde homojen veri setleri üzerinde daha iyi çalıştığından dolayı standartlaşma işlemi uygulanması gerekiyor
# In[62]:


scaler= StandardScaler()


# In[63]:


scaler.fit(x_train)
x_train= scaler.transform(x_train)


# In[64]:


scaler.fit(x_test)
x_test= scaler.transform(x_test)


# In[30]:


mlpc_model= MLPClassifier().fit(x_train,y_train)


# In[31]:


mlpc_model


# In[33]:


mlpc_model.coefs_


# In[34]:


get_ipython().run_line_magic('pinfo', 'mlpc_model')


# In[35]:


y_pred= mlpc_model.predict(x_test)


# In[36]:


accuracy_score(y_test,y_pred)


# In[39]:


mlpc_params= {"hidden_layer_sizes":[(10,10),(100,100,100),(100,100),(3,5)],
             "alpha":[1,5,0.1,0.01,0.03,0.05,0.0001]}


# In[49]:


mlpc= MLPClassifier(solver="lbfgs",activation= "logistic") # sınıflandırma problemlerinde activationu logistic yapmamız gerekiyor 


# In[50]:


mlpc_cv_model= GridSearchCV(mlpc,mlpc_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[51]:


mlpc_cv_model


# In[52]:


mlpc_cv_model.best_params_


# In[152]:


mlpc_tuned= MLPClassifier(solver="lbfgs",alpha=5,hidden_layer_sizes=(100,100),activation= "logistic").fit(x_train,y_train)


# In[59]:


y_pred= mlpc_tuned.predict(x_test)


# In[60]:


accuracy_score(y_test,y_pred)


# # CART

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20212548.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20212548.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20212708.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-09%20212708.png)
libraries used in this field
# In[72]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[67]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")
y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42);


# In[76]:


cart_model= DecisionTreeClassifier().fit(x_train,y_train)


# In[77]:


cart_model


# In[70]:


y_pred= cart_model.predict(x_test)


# In[71]:


accuracy_score(y_test,y_pred)


# In[80]:


cart= DecisionTreeClassifier()


# In[82]:


cart_params= {"max_depth": [1,3,5,8,10],
             "min_samples_split":[2,3,5,10,20,50]}


# In[83]:


cart_cv_model= GridSearchCV(cart,cart_params,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[84]:


cart_cv_model.best_params_


# In[125]:


cart_tuned= DecisionTreeClassifier(max_depth=5,min_samples_split=50).fit(x_train,y_train)


# In[92]:


y_pred= cart_tuned.predict(x_test)


# In[93]:


accuracy_score(y_test,y_pred)


# # Random Forests

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-14%20201726.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-14%20201726.png)![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-14%20201837.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-14%20201837.png)![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-14%20201642.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-14%20201642.png)
libraries used in this field
# In[158]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[4]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")
y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)


# In[5]:


rf_model= RandomForestClassifier().fit(x_train,y_train)


# In[6]:


rf_model


# In[13]:


y_pred= rf_model.predict(x_test)


# In[16]:


accuracy_score(y_test, y_pred)


# In[17]:


rf= RandomForestClassifier()


# In[18]:


rf_params= {"n_estimators":[100,200,500,1000],
           "max_features":[3,5,7,8],
           "min_samples_split": [2,5,10,20]}


# In[19]:


rf_cv_model= GridSearchCV(rf,rf_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[20]:


rf_cv_model.best_params_


# In[126]:


rf_tuned= RandomForestClassifier(n_estimators=200,max_features=5,min_samples_split=5).fit(x_train,y_train)


# In[6]:


y_pred= rf_tuned.predict(x_test)


# In[7]:


accuracy_score(y_test,y_pred)


# In[8]:


# değişken önem düzeyi


# In[9]:


rf_tuned.feature_importances_


# In[11]:


feature= pd.Series(rf_tuned.feature_importances_,
                 index=x_train.columns).sort_values(ascending=False)

sns.barplot(x=feature, y=feature.index)
plt.xlabel('Değişken Önem Skorlarrı')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()


# # Gradient Boosting Machines (GBM)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20141324.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20141324.png)
libraries used in this field
# In[157]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[12]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")
y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)


# In[13]:


gbm_model= GradientBoostingClassifier().fit(x_train,y_train)


# In[14]:


gbm_model


# In[15]:


get_ipython().run_line_magic('pinfo', 'gbm_model')


# In[16]:


y_pred= gbm_model.predict(x_test)


# In[17]:


accuracy_score(y_test,y_pred)


# In[18]:


gbm= GradientBoostingClassifier()


# In[19]:


gbm_params= {"learning_rate":[0.1,0.01,0.001,0.05],
            "n_estimators":[100,300,500,1000],
            "max_depth":[2,3,5,8]}


# In[ ]:


gbm_cv_model= GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[ ]:


gbm_cv_model.best_params_


# In[127]:


gbm_tuned= GradientBoostingClassifier(learning_rate=0.01,
                                      n_estimators=5,
                                      max_depth=500).fit(x_train,y_train)


# In[38]:


y_pred= gbm_tuned.predict(x_test)


# In[39]:


accuracy_score(y_test,y_pred)


# In[40]:


# değişken önem düzeyi


# In[41]:


feature= pd.Series(gbm_tuned.feature_importances_,
                 index=x_train.columns).sort_values(ascending=False)

sns.barplot(x=feature, y=feature.index)
plt.xlabel('Değişken Önem Skorlarrı')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()


# # XGBoost

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20143950.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20143950.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20144027.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20144027.png)
libraries used in this field
# In[43]:


get_ipython().system('pip insatll xgboost')


# In[156]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[42]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")
y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)


# In[46]:


xgb_model= XGBClassifier().fit(x_train,y_train)


# In[47]:


get_ipython().run_line_magic('pinfo', 'xgb_model')


# In[52]:


y_pred=xgb_model.predict(x_test)


# In[53]:


accuracy_score(y_test,y_pred)


# In[54]:


xgb= XGBClassifier()


# In[57]:


xgb_params= {"n_estimators":[100,500,1000],
            "subsample":[0.6,0.8,1],
            "max_depth":[3,5,7],
            "learning_rate":[0.1,0.001,0.01]}


# In[58]:


xgb_cv_model= GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[59]:


xgb_cv_model.best_params_


# In[128]:


xgb_tuned= XGBClassifier(learning_rate=0.01 ,
                        max_depth=3 ,
                        n_estimators=500 ,
                        subsample= 0.8).fit(x_train,y_train)


# In[61]:


y_pred= xgb_tuned.predict(x_test)


# In[62]:


accuracy_score(y_test,y_pred)


# In[65]:


feature= pd.Series(xgb_tuned.feature_importances_,
                 index=x_train.columns).sort_values(ascending=False)

sns.barplot(x=feature, y=feature.index)
plt.xlabel('Değişken Önem Skorlarrı')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()


# # Light GBM

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20153837.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20153837.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20154023.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20154023.png)
libraries used in this field
# In[68]:


get_ipython().system('pip insatll lightgbm')


# In[155]:


from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[67]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")
y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)


# In[97]:


lgbm_model= LGBMClassifier().fit(x_train,y_train)


# In[73]:


lgbm_model


# In[74]:


get_ipython().run_line_magic('pinfo', 'lgbm_model')


# In[75]:


y_predit= lgbm_model.predict(x_test)


# In[76]:


accuracy_score(y_test,y_pred)


# In[77]:


lgbm= LGBMClassifier()


# In[78]:


lgbm_params= {"learnin_rate":[0.001,0.01,0.01],
             "n_estimators":[200,500,100],
             "max_depth":[1,2,35,8]}


# In[79]:


lgbm_cv_model= GridSearchCV(lgbm,lgbm_params,n_jobs=-1,verbose=2,cv=10).fit(x_train,y_train)


# In[80]:


lgbm_cv_model.best_params_


# In[129]:


lgbm_tuned= LGBMClassifier(learning_rate=0.01,
                          max_depth=1,
                          n_estimators=500).fit(x_train,y_train)


# In[86]:


y_pred= lgbm_tuned.predict(x_test)


# In[87]:


accuracy_score(y_test,y_pred)


# In[88]:


feature= pd.Series(lgbm_tuned.feature_importances_,
                 index=x_train.columns).sort_values(ascending=False)

sns.barplot(x=feature, y=feature.index)
plt.xlabel('Değişken Önem Skorlarrı')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()


# # Category Boosting (CatBoost)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20155734.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20155734.png)

# ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20160045.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202024-03-15%20160045.png)
libraries used in this field
# In[90]:


get_ipython().system('pip insatll catboost')


# In[ ]:


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# In[89]:


data= pd.read_csv("C:\\data_set\\diabetes.csv")
y = data["Outcome"]
x= data.drop(["Outcome"],axis = 1)
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.30,random_state=42)


# In[103]:


catb_model= CatBoostClassifier().fit(x_train,y_train,verbose=False)


# In[104]:


y_pred= catb_model.predict(x_test)


# In[105]:


accuracy_score(y_test,y_pred)


# In[110]:


catb= CatBoostClassifier(verbose=False)


# In[111]:


catb_params= {"iterations":[200,500,1000],
             "learning_rate": [0.01,0.03,0.1],
             "depth": [4,5,8]}


# In[112]:


# catb_cv_model= GridSearchCV(catb,catb_params,cv=5,n_jobs=-1,verbose=2).fit(x_train,y_train)


# In[114]:


# catb_cv_model.best_params_


# In[130]:


catb_tuned= CatBoostClassifier(depth=8,
                              iterations=200,
                              learning_rate=0.03).fit(x_train,y_train)


# In[116]:


y_pred= catb_tuned.predict(x_test)


# In[118]:


accuracy_score(y_test,y_pred)


# In[119]:


feature= pd.Series(catb_tuned.feature_importances_,
                 index=x_train.columns).sort_values(ascending=False)

sns.barplot(x=feature, y=feature.index)
plt.xlabel('Değişken Önem Skorlarrı')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()


# # Tüm Modellerin Karşılaştırılması

# In[153]:


modeller =[
    knn_tuned,
    loj_model,
    svm_tuned,
    mlpc_tuned,
    cart_tuned,
    rf_tuned,
    gbm_tuned,
    catb_tuned,
    lgbm_tuned,
    xgb_tuned]
sonuc= []
sonuclar = pd.DataFrame(columns=["Modeller", "Accuracy"])

for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(x_test)
    dogruluk = accuracy_score(y_test, y_pred)
    sonuc = pd.DataFrame([[isimler, dogruluk * 100]], columns=["Modeller", "Accuracy"])
    sonuclar = pd.concat([sonuclar, sonuc], ignore_index=True)


# In[149]:


sonuclar = sonuclar.sort_values(by='Accuracy', ascending=False)

sns.barplot(x='Accuracy',y='Modeller',data= sonuclar, color="r")
plt.xlabel('Accuracy %')
plt.title('Modellerin Doğruluk Oranları');


# In[143]:


sonuclar

yüksek korelasyonu sahip değişkenler den bazıları çıkarılır çünkü benzer şeyler açıklamaktadırlar
# In[160]:


get_ipython().run_line_magic('pinfo', 'gbm_model')


# In[161]:


# Logistic Regression
loj_model = LogisticRegression(solver="liblinear").fit(x,y)

# K-EN Yakın Komşu
knn_model= KNeighborsClassifier().fit(x_train,y_train)
knn_params= {"n_neighbors":np.arange(1,50)}

# Support Vector Machines (SVM)
svm_model= SVC(kernel= "linear").fit(x_train,y_train)
svm_params= {"C":np.arange(1,5),
            "kernel":["linear","rbf"]}

# Artificial neural networks
scaler= StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
scaler.fit(x_test)
x_test= scaler.transform(x_test)
mlpc_model= MLPClassifier().fit(x_train,y_train)
mlpc_params= {"hidden_layer_sizes":[(10,10),(100,100,100),(100,100),(3,5)],
             "alpha":[1,5,0.1,0.01,0.03,0.05,0.0001]}
mlpc= MLPClassifier(solver="lbfgs",activation= "logistic") # sınıflandırma problemlerinde activationu logistic yapmamız gerekiyor 


# CART
cart_model= DecisionTreeClassifier().fit(x_train,y_train)
cart_params= {"max_depth": [1,3,5,8,10],
             "min_samples_split":[2,3,5,10,20,50]}


# Random Forest
rf_model= RandomForestClassifier().fit(x_train,y_train)
rf_params= {"n_estimators":[100,200,500,1000],
           "max_features":[3,5,7,8],
           "min_samples_split": [2,5,10,20]}


# Gradient Boosting Machines (GBM)
gbm_model= GradientBoostingClassifier().fit(x_train,y_train)
gbm_params= {"learning_rate":[0.1,0.01,0.001,0.05],
            "n_estimators":[100,300,500,1000],
            "max_depth":[2,3,5,8]}

# XGBoost
xgb_model= XGBClassifier().fit(x_train,y_train)
xgb_params= {"n_estimators":[100,500,1000],
            "subsample":[0.6,0.8,1],
            "max_depth":[3,5,7],
            "learning_rate":[0.1,0.001,0.01]}

# Light GBM
lgbm_model= LGBMClassifier().fit(x_train,y_train)
lgbm_params= {"learnin_rate":[0.001,0.01,0.01],
             "n_estimators":[200,500,100],
             "max_depth":[1,2,35,8]}


# Category Boosting (CatBoost)
catb_model= CatBoostClassifier().fit(x_train,y_train,verbose=False)
catb_params= {"iterations":[200,500,1000],
             "learning_rate": [0.01,0.03,0.1],
             "depth": [4,5,8]}


# In[ ]:




