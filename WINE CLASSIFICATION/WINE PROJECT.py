#!/usr/bin/env python
# coding: utf-8

# # CLASSIFICATION ALGORITHMS

# #### Definition:
# In machine learning and statistics, classification is the problem of identifying to which of a set of categories a new observation belongs, on the basis of a training set of data containing observations whose category membership is known.

# ## WINE QUALITY DATA

# ### Description of the dataset:
# The dataset is related to the red variant of Portuguese "Vinho Verde" wine which can be viewed as a classification data.Due to privacy only the physiochemical (inputs) and sensory (output) variables are available.The datset has a total of 11 attributes/variables plus an output variable(class) with 1599 instances/observations.The classes are ordered but not balanced.

# ### Attribute Information:
# Input variables (based on physicochemical tests):\
#    1 - fixed acidity\
#    2 - volatile acidity\
#    3 - citric acid\
#    4 - residual sugar\
#    5 - chlorides\
#    6 - free sulfur dioxide\
#    7 - total sulfur dioxide\
#    8 - density\
#    9 - pH\
#    10 - sulphates\
#    11 - alcohol\
#    Output variable (based on sensory data): \
#    12 - quality (score between 0 and 10)

# In[1]:


# Importing necessary packages and functions required
import numpy as np # for numerical computations
import pandas as pd # for data processing,I/O file operations
import matplotlib.pyplot as plt # for visualization of different kinds of plots
get_ipython().run_line_magic('matplotlib', 'inline')
# for matplotlib graphs to be included in the notebook, next to the code
import seaborn as sns # for visualization 
import warnings # to silence warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing red wine data into a dataframe
data=pd.read_csv("D:\\20-6-19\\PROJECTS\\WINE CLASSIFICATION\\winequality-red.csv")


# In[3]:


# Glimpse of the data
data.sample(5)


# In[4]:


#shape of the data i.e., no of rows and columns in the data
data.shape


# In[5]:


#size of the data
data.size


# ### Data Analysis and Visualization

# In[6]:


# data information i.e., datatypes of different columns,their count etc
data.info()


# In[7]:


# Description of the data i.e., Descriptive Statistics
data.describe()


# In[8]:


# checking the different classes of the wine quality 
data.quality.unique()


# We observe there are a total of 6 unique wine qualities in our data.

# In[9]:


# Checking the number of supporting observations for each class of wine quality
data['quality'].value_counts()


# The wine quality 5 has the maximum supporting cases in the data of 681 cases,while the wine qualities 3,8 have very less number of supporting cases of 10,18 respectively. 

# In[10]:


sns.countplot(data.quality)
plt.show()


# From the above count plot we find that wines with normal quality(4,5,6,7) have more no of instances while the excellent or poor quality wines(8,3) respectively have less instances for support.

# Since the classes are not balanced,we remove the classes with less supporting classes i.e, qualities 3, 8 as they hinder the learning process of the models while fitting the data which produces abnormal results.

# In[11]:


# Get names of indexes for which column Age has value 30
indexNames1 = data[ (data['quality'] == 3) ].index
indexNames2 = data[ (data['quality'] == 8) ].index
 
# Delete these row indexes from dataFrame
data.drop(indexNames1, inplace=True)
data.drop(indexNames2,inplace=True)


# In[12]:


# Checking the number of supporting observations for each class of wine quality
data['quality'].value_counts()


# In[13]:


sns.countplot(data.quality)
plt.show()


# Now each class have a decent number of supporting classes for the model to learn and classify a new one.

# In[14]:


# Now lets see the shape of the data
data.shape


# In[15]:


# Checking for missing values in the data
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap="viridis")
plt.show()


# From the above heat map we observe that there are no missing values in the data.

# In[16]:


# Checking fixed acidity levels for each wine quality
fig = plt.figure(figsize = (8,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = data)


# In[17]:


# Takes more run time can avoid this code if considered unnecessary.
fig,ax=plt.subplots(4,2,figsize=(15,15))
plt.subplots_adjust(hspace=.4)
ax[0,0].bar(x='quality',height='fixed acidity',data = data)
ax[0,1].bar(x="quality",height="volatile acidity",data=data)
ax[1,0].bar(x="quality",height="citric acid",data=data)
ax[1,1].bar(x="quality",height="residual sugar",data=data)
ax[2,0].bar(x="quality",height="chlorides",data=data)
ax[2,1].bar(x="quality",height="free sulfur dioxide",data=data)
ax[3,0].bar(x="quality",height="sulphates",data=data)
ax[3,1].bar(x="quality",height="alcohol",data=data)
ax[0,0].set_title("fixed acidity")
ax[0,1].set_title("volatile acidity")
ax[1,0].set_title("citric acid")
ax[1,1].set_title("residual sugar")
ax[2,0].set_title("chlorides")
ax[2,1].set_title("free sulfur dioxide")
ax[3,0].set_title("sulphates")
ax[3,1].set_title("alcohol")
plt.show()


# We can see various levels of different features(fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,sulphates,alcohol) for different kinds of wine quality.

# In[18]:


fig = plt.figure(figsize = (9,6))
sns.pointplot(x=data['pH'].round(1),y='residual sugar',color='green',data=data)
plt.show()


# From the above point plot we can see various point estimates and confidence levels for residual sugar levels at different values of pH.

# In[19]:


fig = plt.figure(figsize = (8,6))
sns.pointplot(y=data['pH'].round(1),x='quality',color='MAGENTA',data=data)
plt.show()


# At different wine qualities the point estimates and confidence intervals for pH values are shown in the above point plot.

# In[20]:


# Takes more run time can avoid this code if considered unnecessary. 
sns.pairplot(data)
plt.show()


# In[21]:


corr=data.corr()


# In[22]:


corr


# In[23]:


# Visualizing correlation
plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True)
plt.show()


# ## FITTING MODELS TO THE DATASET

# ### Models applying:

# 
# Logistic regression\
# Linear SVM\
# rbf SVM\
# KNN\
# Gaussian NB\
# Decision Tree\
# Random Forest\
# Gradient Boosting
# 

# #### SPLITTING X AND Y VARIABLES

# In[24]:


# SPLITING X AND Y VARIABLES
X=data.iloc[:,:-1]
y=data.iloc[:,11]


# In[25]:


# A GLIMPSE OF X AND Y VARAIBLES
X.sample(3)


# In[26]:


y.sample(3)


# #### TRAINING AND TESTING DATASETS

# In[27]:


# SPLITTING DATASET INTO TRAINING AND TESTING DATA
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[28]:


# SHAPE OF TRAINING AND TESTING DATA
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### FITTING ALL MODELS AT THE SAME TIME

# In[29]:


# Importing packages and functions required for fitting different models to the data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


# Importing functions to get the model fitting for the data 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix  
from sklearn.model_selection import cross_val_score


# ####  Fitting all the models at the same time using 'for' loop and obtaining Confusion matrices,Accuracies

# In[30]:


# Fitting all the models at the same time using 'for' loop
models=[LogisticRegression(multi_class="multinomial",solver="newton-cg"),
        LinearSVC(),
        SVC(kernel='rbf',gamma="auto"),
        KNeighborsClassifier(n_neighbors=10,metric="euclidean"),
        GaussianNB(),
        DecisionTreeClassifier(criterion="gini",max_depth=10),
        RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier()
        ]
model_names=['LogisticRegression',
             'LinearSVM',
             'rbfSVM',
             'KNearestNeighbors',
             'GaussianNB',
             'DecisionTree',
             'RandomForestClassifier',
             'GradientBoostingClassifier',
             ]
acc=[]

for model in range(len(models)):
    classification_model=models[model]
    classification_model.fit(X_train,y_train)
    y_pred=classification_model.predict(X_test)
    acc.append(accuracy_score(y_pred,y_test))
    print("confusion matrix of:",model_names[model],"\n",confusion_matrix(y_test,y_pred))  
d={'Modelling Algorithm':model_names,'Accuracy':acc}
acc_table=pd.DataFrame(d)
acc_table


# TABLE SHOWING EACH MODEL AND ITS CORRESPONDING ACCURACY SCORES.

# In[31]:


sns.barplot(y='Modelling Algorithm',x='Accuracy',data=acc_table)


# Bar plot representing the acuracies of all the models applied.

# In[32]:


sns.catplot(x='Modelling Algorithm',y='Accuracy',data=acc_table,kind='point',height=4,aspect=3.5)
plt.show()


# Cat plot representing accuracies for different models applied.

# ### Finding 10 fold cross validation scores for all the models at the same time using 'for' loop 

# In[33]:


# Finding 10 fold cross validation scores for all the models at the same time using 'for' loop 
models=[LogisticRegression(multi_class="multinomial",solver="newton-cg"),
        LinearSVC(),
        SVC(kernel='rbf',gamma="auto"),
        KNeighborsClassifier(n_neighbors=10,metric="euclidean"),
        GaussianNB(),
        DecisionTreeClassifier(criterion="gini",max_depth=10),
        RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier()
        ]
model_names=['LogisticRegression',
             'LinearSVM',
             'rbfSVM',
             'KNearestNeighbors',
             'GaussianNB',
             'DecisionTree',
             'RandomForestClassifier',
             'GradientBoostingClassifier',
             ]
cvs=[]
for model in range(len(models)):
    classification_model=models[model]
    clf=classification_model.fit(X_train,y_train)
    scores = cross_val_score(clf, X_test, y_test, cv=10)
    scores.mean()
    print("10 fold cross validation of:",model_names[model],"\n",scores.mean())  


# ### PREDICTING A NEW OBSERVATION WITH ALL DIFFERENT MODELS WHOSE ACTUAL VALUE IS 5.

# In[34]:


# prediction 
new_obs=[[9,0.580,0.25,2.8,0.075,9.0,104.0,0.99779,3.23,0.57,9.7]]
pv=[]
for model in range(len(models)):
    classification_model=models[model]
    models[model].predict(new_obs)
    pv.append(models[model].predict(new_obs))
    
d={'Modelling Algorithm':model_names,'Predicted value':pv}
pred_table=pd.DataFrame(d)
pred_table


# ## **Conclusion:**

# FROM ALL THE ABOVE VALUES WE SEE THAT **SUPPORT VECTOR MACHINE(RBF)**  PREDICTED WRONGLY.
