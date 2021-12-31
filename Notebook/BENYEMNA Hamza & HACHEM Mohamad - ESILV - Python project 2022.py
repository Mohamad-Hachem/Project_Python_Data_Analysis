#!/usr/bin/env python
# coding: utf-8

# # I) Dataset
# 
# <ins>Data set can be found:</ins>
# * https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

# ## 1) General context : problematic and target of the project
# 
# Our dataset is focusing on statistics concerning diabetics in the US from 1999 to 2008.
# The patients affected by diabetes are treated by some hospitals. In order to treat them in an efficient way, the hospitals try to make hypothesis on the readmission of the patients according to some features such as the health background of each patient.
# 
# Indeed, the hospitals settle the readmissions of the patients in 3 ways:
#  - Those that have been readmitted in less than 30 days
#  - Those that have been readmitted in more than 30 days
#  - Those that have not been readmitted at all
#  
# The purpose of this project is to bring an efficient model of prediction that could separate the patient into 3 groups as mentionned (< 30 days, > 30 days and no readmission).
# 
# This is a 3-class classification problem. Moreover, we have a dataset that could train our model. Therefore, it is a supervised model that could be trained.

# ## 2) Introduction to the dataset

# ### Importation of libraries
# 
# We need the following packages in order to import the dataset, work on it and show some plots if required.

# In[3]:


import pandas as pd
import numpy as np


# ### Importation of the dataset

# In[4]:


df = pd.read_csv("diabetic_dat.csv")
df


# We have a dataset of 101766 registred diabetics according 50 features.

# ### Description of the dataset

# In[5]:


df.head()


# In[6]:


df.info()


# By checking the info, we can conclude few things:
# 
# * The **readmitted** column is the column will be using to know the classification (this is the target).
# * The type of our features are <ins>integer</ins> or <ins>object</ins> (<ins>string</ins>).
# 

# ### Names of features

# In[7]:


print(list(df))


# From now, we have an idea of the dataset.
# The issues that we have to face with could be:
# 
#  - The **lake of data** ("?" and "Nan" for example)
#  - **heterogeneous data** (<ins>integers</ins> and <ins>strings</ins> in the same column)
#  - **imprecise data** (the age registred by <ins>intervals</ins>)
#  
# That is why it could be better to **clean** and **analyze** the dataset in order the exploit it in the most efficient way.

# ## 3) Data Cleaning

# ### Duplicates

# First, we have noticed duplicated data. We could remove it.
# 
# We will use **patient_nbr** as a reference since it has a unique numbers.

# In[8]:


df['patient_nbr'].value_counts()


# Out of **101,766** we find that only **71518** are unique. <br>
# Let's try to keep the unique records.

# In[9]:


df = df.drop_duplicates(subset=['patient_nbr'])
df


# ### Useless features

# Let's run some tests on our different features and see if some columns can be removed.

# <ins>Weight</ins>

# In[10]:


numberPresentRows =  df.loc[(df['weight'] != '?'),'weight']
numberMissingRows = len(df.index)
print(len(numberPresentRows))


# Only 2853 rows have no missing weight.

# In[11]:


numberPresentRows =  df.loc[(df['weight'] != '?')].weight.count()
numberMissingRows = len(df.index)
print("percentage of missing rows of {} {} %".format("weight",100-((numberPresentRows/numberMissingRows)*100)//1))


# We can conclude that the **weight** feature is almost negligible because of its amount of missing data representing 97% of the dataset.<br>

# <ins>Drugs</ins>

# In[12]:


drugslist = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
for i in range(len(drugslist)):
    numberPresentRows =  len(df.loc[(df[drugslist[i]] != 'No'),drugslist[i]])
    numberMissingRows = len(df.index) 
    print( "percentage of patients not using the drug of {} {} %".format(drugslist[i],100-((numberPresentRows/numberMissingRows)*100)//1))


# We can see that a lot of the drugs are included in our dataset yet not really used by the patients <br>
# meaning that it doesn't really affect the our target **readdmited** column.<br>
# 
# We decide remove every drug that has less that < 5% impact.
# 
# We will also remove every column that doesn't affect our work directly such as:
#  - incounter_id
#  - patient_nbr
#  - payer_code

# In[13]:


columns_drop_list = ['encounter_id', 'patient_nbr', 'weight', 'payer_code','repaglinide', 'nateglinide', 'chlorpropamide','acetohexamide','tolbutamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone','medical_specialty']
df.drop(columns_drop_list,axis = 1, inplace=True)


# In[14]:


df.info()


# The features that we decided to consider as useless are droped frome our dataset. We could now work on the 27 features left to predict the target.

# ### Mapping the dataset

# In order to manipulate and work with our data more easily, we decided to **map** the data.
# 
# That is to say, we assign <ins>numeric values</ins> to all the data.

# **Diag**

# We replace <ins>"V"</ins> or <ins>"E"</ins> by <ins>"0"</ins>.

# In[15]:


#start by setting all values containing E or V into 0 (as one category)
df.loc[df['diag_1'].str.contains('V',na=False,case=False), 'diag_1'] = 0
df.loc[df['diag_1'].str.contains('E',na=False,case=False), 'diag_1'] = 0
df.loc[df['diag_2'].str.contains('V',na=False,case=False), 'diag_2'] = 0
df.loc[df['diag_2'].str.contains('E',na=False,case=False), 'diag_2'] = 0
df.loc[df['diag_3'].str.contains('V',na=False,case=False), 'diag_3'] = 0
df.loc[df['diag_3'].str.contains('E',na=False,case=False), 'diag_3'] = 0


# The missing data is replace by <ins>"-1"</ins>.

# In[16]:


#setting all missing values into -1
df['diag_1'] = df['diag_1'].replace('?', -1)
df['diag_2'] = df['diag_2'].replace('?', -1)
df['diag_3'] = df['diag_3'].replace('?', -1)


# Then , we assign a <ins>unique</ins> value to intervals.

# In[17]:


#No all diag values can be converted into numeric values
df['diag_1'] = df['diag_1'].astype(float)
df['diag_2'] = df['diag_2'].astype(float)
df['diag_3'] = df['diag_3'].astype(float)

df['diag_1'].loc[(df['diag_1']>=1) & (df['diag_1']< 140)] = 1
df['diag_1'].loc[(df['diag_1']>=140) & (df['diag_1']< 240)] = 2
df['diag_1'].loc[(df['diag_1']>=240) & (df['diag_1']< 280)] = 3
df['diag_1'].loc[(df['diag_1']>=280) & (df['diag_1']< 290)] = 4
df['diag_1'].loc[(df['diag_1']>=290) & (df['diag_1']< 320)] = 5
df['diag_1'].loc[(df['diag_1']>=320) & (df['diag_1']< 390)] = 6
df['diag_1'].loc[(df['diag_1']>=390) & (df['diag_1']< 460)] = 7
df['diag_1'].loc[(df['diag_1']>=460) & (df['diag_1']< 520)] = 8
df['diag_1'].loc[(df['diag_1']>=520) & (df['diag_1']< 580)] = 9
df['diag_1'].loc[(df['diag_1']>=580) & (df['diag_1']< 630)] = 10
df['diag_1'].loc[(df['diag_1']>=630) & (df['diag_1']< 680)] = 11
df['diag_1'].loc[(df['diag_1']>=680) & (df['diag_1']< 710)] = 12
df['diag_1'].loc[(df['diag_1']>=710) & (df['diag_1']< 740)] = 13
df['diag_1'].loc[(df['diag_1']>=740) & (df['diag_1']< 760)] = 14
df['diag_1'].loc[(df['diag_1']>=760) & (df['diag_1']< 780)] = 15
df['diag_1'].loc[(df['diag_1']>=780) & (df['diag_1']< 800)] = 16
df['diag_1'].loc[(df['diag_1']>=800) & (df['diag_1']< 1000)] = 17
df['diag_1'].loc[(df['diag_1']==-1)] = 0

df['diag_2'].loc[(df['diag_2']>=1) & (df['diag_2']< 140)] = 1
df['diag_2'].loc[(df['diag_2']>=140) & (df['diag_2']< 240)] = 2
df['diag_2'].loc[(df['diag_2']>=240) & (df['diag_2']< 280)] = 3
df['diag_2'].loc[(df['diag_2']>=280) & (df['diag_2']< 290)] = 4
df['diag_2'].loc[(df['diag_2']>=290) & (df['diag_2']< 320)] = 5
df['diag_2'].loc[(df['diag_2']>=320) & (df['diag_2']< 390)] = 6
df['diag_2'].loc[(df['diag_2']>=390) & (df['diag_2']< 460)] = 7
df['diag_2'].loc[(df['diag_2']>=460) & (df['diag_2']< 520)] = 8
df['diag_2'].loc[(df['diag_2']>=520) & (df['diag_2']< 580)] = 9
df['diag_2'].loc[(df['diag_2']>=580) & (df['diag_2']< 630)] = 10
df['diag_2'].loc[(df['diag_2']>=630) & (df['diag_2']< 680)] = 11
df['diag_2'].loc[(df['diag_2']>=680) & (df['diag_2']< 710)] = 12
df['diag_2'].loc[(df['diag_2']>=710) & (df['diag_2']< 740)] = 13
df['diag_2'].loc[(df['diag_2']>=740) & (df['diag_2']< 760)] = 14
df['diag_2'].loc[(df['diag_2']>=760) & (df['diag_2']< 780)] = 15
df['diag_2'].loc[(df['diag_2']>=780) & (df['diag_2']< 800)] = 16
df['diag_2'].loc[(df['diag_2']>=800) & (df['diag_2']< 1000)] = 17
df['diag_2'].loc[(df['diag_2']==-1)] = 0

df['diag_3'].loc[(df['diag_3']>=1) & (df['diag_3']< 140)] = 1
df['diag_3'].loc[(df['diag_3']>=140) & (df['diag_3']< 240)] = 2
df['diag_3'].loc[(df['diag_3']>=240) & (df['diag_3']< 280)] = 3
df['diag_3'].loc[(df['diag_3']>=280) & (df['diag_3']< 290)] = 4
df['diag_3'].loc[(df['diag_3']>=290) & (df['diag_3']< 320)] = 5
df['diag_3'].loc[(df['diag_3']>=320) & (df['diag_3']< 390)] = 6
df['diag_3'].loc[(df['diag_3']>=390) & (df['diag_3']< 460)] = 7
df['diag_3'].loc[(df['diag_3']>=460) & (df['diag_3']< 520)] = 8
df['diag_3'].loc[(df['diag_3']>=520) & (df['diag_3']< 580)] = 9
df['diag_3'].loc[(df['diag_3']>=580) & (df['diag_3']< 630)] = 10
df['diag_3'].loc[(df['diag_3']>=630) & (df['diag_3']< 680)] = 11
df['diag_3'].loc[(df['diag_3']>=680) & (df['diag_3']< 710)] = 12
df['diag_3'].loc[(df['diag_3']>=710) & (df['diag_3']< 740)] = 13
df['diag_3'].loc[(df['diag_3']>=740) & (df['diag_3']< 760)] = 14
df['diag_3'].loc[(df['diag_3']>=760) & (df['diag_3']< 780)] = 15
df['diag_3'].loc[(df['diag_3']>=780) & (df['diag_3']< 800)] = 16
df['diag_3'].loc[(df['diag_3']>=800) & (df['diag_3']< 1000)] = 17
df['diag_3'].loc[(df['diag_3']==-1)] = 0


# In[15]:


df[['diag_1','diag_2','diag_3']]


# We can see that our numerization works.
# 
# Let's do it with the other features that are non numerics values.
# 
# 

# **Race**
# 
# We assign:
#  - <ins>"0"</ins> to <ins>"Caucasian"</ins>
#  - <ins>"1"</ins> to <ins>"AfricanAmerican"</ins>
#  - <ins>"2"</ins> to <ins>"Hispanic"</ins>
#  - <ins>"3"</ins> to <ins>"Asian"</ins>
#  - <ins>"4"</ins> to <ins>"Other"</ins>

# In[93]:


df.race.value_counts()


# In[18]:


df['race'] = df['race'].replace('?','Other')
df.race.value_counts()


# In[19]:


df['race'] = df['race'].replace('Caucasian', 0)
df['race'] = df['race'].replace('AfricanAmerican', 1)
df['race'] = df['race'].replace('Hispanic', 2)
df['race'] = df['race'].replace('Asian', 3)
df['race'] = df['race'].replace('Other', 4)
df.race.value_counts()


# **Gender**
# 
# We assign:
#  - <ins>"0"</ins> to <ins>"Female"</ins>
#  - <ins>"1"</ins> to <ins>"Male"</ins>
#  
# In order to simplify the work, we will consider <ins>"Unknown/Invalid"</ins> as <ins>"Female"</ins>

# In[20]:


df['gender'] = df['gender'].replace('Unknown/Invalid', 'Female')
df.gender.value_counts()


# In[21]:


df['gender'] = df['gender'].replace('Male', 1)
df['gender'] = df['gender'].replace('Female', 0)
df.gender.value_counts()


# **Age**
# 
# The age is given by intervals.
# 
# So, we decided to take the average for each interval and to assign it.

# In[22]:


df.age.value_counts()


# In[23]:


#we will take the middle of every interval for example [0-10] -> 5
for i in range(0,10):
    df['age'] = df['age'].replace('['+str(10*i)+'-'+str(10*(i+1))+')', i*10+5)
df['age'].value_counts()


# **Max_glu_serum**
# 
# We assign:
#  - <ins>"0"</ins> to <ins>"None"</ins>
#  - <ins>"1"</ins> to <ins>"Norm"</ins>
#  - <ins>"2"</ins> to <ins>">200"</ins>
#  - <ins>"3"</ins> to <ins>">300"</ins>

# In[22]:


df.max_glu_serum.value_counts()


# In[24]:


df['max_glu_serum']=df['max_glu_serum'].replace("None", 0)
df['max_glu_serum']=df['max_glu_serum'].replace("Norm", 1)
df['max_glu_serum']=df['max_glu_serum'].replace(">200", 2)
df['max_glu_serum']=df['max_glu_serum'].replace(">300", 3)
df.max_glu_serum.value_counts()


# **A1Cresult**
# 
# We assign:
#  - <ins>"0"</ins> to <ins>"None"</ins>
#  - <ins>"1"</ins> to <ins>"Norm"</ins>
#  - <ins>"2"</ins> to <ins>">7"</ins>
#  - <ins>"3"</ins> to <ins>">8"</ins>

# In[24]:


df.A1Cresult.value_counts()


# In[25]:


df['A1Cresult']=df['A1Cresult'].replace("None", 0)
df['A1Cresult']=df['A1Cresult'].replace("Norm", 1)
df['A1Cresult']=df['A1Cresult'].replace(">7", 2)
df['A1Cresult']=df['A1Cresult'].replace(">8", 3)
df.A1Cresult.value_counts()


# **Drugs**
# 
# We assign:
#  - <ins>"0"</ins> to <ins>"No"</ins>
#  - <ins>"1"</ins> to <ins>"Down"</ins>
#  - <ins>"2"</ins> to <ins>"Steady"</ins>
#  - <ins>"3"</ins> to <ins>"Up"</ins>

# In[26]:


df.insulin.value_counts()


# In[26]:


#we will work with the drugs that we decided to keep into our work
drug_list = ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin']
for i in drug_list:
    df[i] = df[i].replace('No', 0)
    df[i] = df[i].replace('Steady', 2)
    df[i] = df[i].replace('Down', 1)
    df[i] = df[i].replace('Up', 3)
df.insulin.value_counts()


# **Change**
# 
# We assign:
#  - <ins>"0"</ins> to <ins>"No"</ins>
#  - <ins>"1"</ins> to <ins>"Ch"</ins>

# In[28]:


df.change.value_counts()


# In[27]:


df['change']=df['change'].replace('No', 0)
df['change']=df['change'].replace('Ch', 1)
df.change.value_counts()


# **DiabetesMed**
# 
# We assign:
#  - <ins>"0"</ins> to <ins>"No"</ins>
#  - <ins>"1"</ins> to <ins>"Yes"</ins>

# In[30]:


df.diabetesMed.value_counts()


# In[28]:


df['diabetesMed']=df['diabetesMed'].replace('Yes', 1)
df['diabetesMed']=df['diabetesMed'].replace('No', 0)
df.diabetesMed.value_counts()


# **Readmitted**
# 
# We assign:
#  - <ins>"0"</ins> to <ins>"No"</ins>
#  - <ins>"1"</ins> to <ins>">30" </ins>
#  - <ins>"2"</ins> to <ins> "<30" </ins>

# In[32]:


df.readmitted.value_counts()


# In[29]:


df['readmitted']=df['readmitted'].replace('NO', 0)
df['readmitted']=df['readmitted'].replace('>30', 1)
df['readmitted']=df['readmitted'].replace('<30', 2)
df.readmitted.value_counts()


# In[30]:


df.info()


# The data is cleaned. From now, we could manipulate it easily and visualize it through **plots** in order to find out some important predictors to fit our models.

# ## 4) Viewing and analysis of the dataset

# ### Importation of librairies

# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# ### Correlation

# Let us see what are the correlation between them in order to pick up only the most valuable features to fit our models.

# In[35]:


matrix = np.triu(df.corr())
fig, ax = plt.subplots(figsize=(24,16))
sns.heatmap(df.corr(), annot=True, ax=ax, fmt='.1g', vmin=-1, vmax=1, center= 0, mask=matrix, cmap='RdBu_r')
plt.show()


# According to our matrix correlation, we can see:
#  - **diabetesMed** is corrolated with **insulin** (0.5)
#  - **num_medications** is corrolated with **time_in_hospital**(0.5)
# 
# But we aim to find the most correlated features with our target, the **readmitted**.
# 
# We notice that our target has a low correlation with all the features (<0.1).
# 
# The most interesting ones are :
#  - **number_inpatient** (0.1)
#  - **number_diagnoses** (0.09)
#  - **number_emergency** (0.07)
#  - **age** (0.07)
#  - **diabetesMed** (0.06)

# ### Plots

# We choose the **gender** and the **race** because they always could be interesting features to analyze.
# 
# Then, we will plot some interesting features that have the highest correlation with the target.

# **Race**

# In[36]:


fig = plt.figure(figsize=(18, 6))
sns.countplot(data=df, x='race', hue='readmitted')
plt.show()


# We can say that readmittion has simmilar distribution almost across different races.

# **Gender**

# In[37]:


fig = plt.figure(figsize=(18, 6))
sns.countplot(data=df, x='gender', hue='readmitted')
plt.show()


# We can see that there is slightly higher female patients. We know that this result is skewed by the **Unknown** patients considered as **Females**.

# **Number_inpatient**

# In[38]:


fig = plt.figure(figsize=(18, 6))
sns.countplot(data=df, x='number_inpatient', hue='readmitted')
plt.show()


# As a matter of fact, patients with 0 inpatient number are not readmitted. Then, the readmission decrease with a higher number of inpatients.

# **Number_diagnoses**

# In[39]:


fig = plt.figure(figsize=(18, 6))
sns.countplot(data=df, x='number_diagnoses', hue='readmitted')
plt.show()


# This plot shows that the number of patients is higher for number of diagnoses 9 and very low for number 1. We can also say that the readmission increase with the number of diagnoses.

# **Number_emergency**

# In[40]:


fig = plt.figure(figsize=(18, 6))

sns.countplot(data=df, x='number_emergency', hue='readmitted')
plt.show()


# The readmission is higher with a lower number of emergency.

# **Age**

# In[41]:


fig = plt.figure(figsize=(18, 6))
sns.countplot(data=df, x='age', hue='readmitted')
plt.show()


# we can see that there are few cases before the age of 40 and they peak around the age of 75.

# **diabetesMed**

# In[42]:


fig = plt.figure(figsize=(18, 6))
sns.countplot(data=df, x='diabetesMed', hue='readmitted')
plt.show()


# We can say that readmitted has the same distribution for the people with or without diabetes medication in terms of proportions between no readmission, < 30 days and > 30 days. But it is interesting to add that the patients that have medications are more concerned by readmission than the ones with no medications (3x more).

# **Time spent in hospital**

# Therefore, there is no correlation that is obvious. But, there are some interesting facts as:
#  - readmission is similar through **races**
#  - **older** patients are more concerned by readmission
#  - **males** are more concerned than **females**
#  - patients with **medications** are more concerned than the others
#  - the lower is the **inpatient number**, the higher there is readmission
#  - the higher is the **number diagnoses**, the higher is the readmission

# According to this, we have a clean dataset and a good analysis in order to apply some interesting models.

# # II) Training a Machine Learning model

# In order to predict the readmitted label, we have many choices.
# 
# The choice depends on:
#  - the lenght of the dataset (more or less than 100 000 rows)
#  - labelled or unlabelled dataset
# 
# We have a labelled dataset with less than 100 000 rows. So, the most adapted models are provided by:
#  - Naive Bayes
#  - SVM
#  - Random Forest
#  - Gradient Boosting
#  
# If we need speed, the most interesting model is **Naive Bayes**.
# 
# If we need accuracy, the most interesting is **Random Forest** and **Gradient Boosting**.

# ### Importation of librairies

# From the librairy sklearn, we have an access to all the methods of model prediction previously introduced.

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# In[33]:


x = df[df.columns[0:df.shape[1]-1]]
y = df['readmitted']
print(y.value_counts())


# ### Data split

# We split the data into 2 parts: a **training set** to fit our model and a **testing set** to evaluate its accuracy.
# 
# <ins>X</ins> represents the **predictors** and <ins>Y</ins> the **target**.

# We decided to allocate **2/3** of the data to the training set and **1/3** to the testing set.

# In[34]:


x_train, x_test , y_train ,y_test = train_test_split(x,y,test_size=0.33)
print("X train: ",x_train.shape)
print("X test: ",x_test.shape)
print("Y train: ",y_train.shape)
print("Y test: ",y_test.shape)


# ### Data scaling

# The scaling helps to have more homogeneous and calibrate numeric values.

# In[35]:


scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(x_test)


# ## 1) Differents models

# Now we will start building our models and see which one is the best.

# **Naive Bayes**

# In[36]:


naive_Bayes_model = GaussianNB()
naive_Bayes_model.fit(x_train,y_train)
naive_Bayes_model_pred = naive_Bayes_model.predict(x_test)


# **Random Forest**

# In[37]:


random_forest_model = RandomForestClassifier()
random_forest_model.fit(x_train,y_train)
random_forest_model_pred = random_forest_model.predict(x_test)


# **Gradient Boosting**

# In[38]:


gradient_boosting_model = GradientBoostingClassifier() 
gradient_boosting_model.fit(x_train,y_train)
gradient_boosting_model_pred = gradient_boosting_model.predict(x_test)


# Now that we built our models, we will see which model works best. Our purpose is to obtain the best **accuracy** possible.

# In[39]:


naive_Bayes_model_accuracy = naive_Bayes_model.score(x_test,y_test)
random_forest_model_accuracy = random_forest_model.score(x_test,y_test)
gradient_boosting_model_accuracy = gradient_boosting_model.score(x_test,y_test)

results = pd.DataFrame({
    'Model':['Naive Bayes','Random Forest','Gradient Boosting'],
    'Score':[naive_Bayes_model_accuracy,random_forest_model_accuracy,gradient_boosting_model_accuracy]
})

results_df = results.sort_values(by='Score',ascending=False)
results_df = results_df.set_index('Score')
results_df


# According to our models, we prefer the **Gradient Boosting** because he has the best accuracy rate : <ins>61.5%</ins>

# ## 2) Model improvement

# To improve our models, we could split our testing set into 2 parts :
#  - **validation set**
#  - **testing set**
# 
# The validation test allow us to verify our model thanks to a score and a **confusion matrix**

# In[40]:


x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_test, y_test, test_size=0.5, random_state=101)


# In[41]:


gradient_boosting_model_pred = gradient_boosting_model.predict(x_val_train)

print('\nGradient Boost initial Performance:')

print('F1 Score        : ', metrics.f1_score(y_val_train, gradient_boosting_model_pred,average='micro'))
print('Confusion Matrix:\n ', confusion_matrix(y_val_train, gradient_boosting_model_pred))


# We can see that the score is about <ins>60-62,5%</ins>

# ### Variables importance

# Every feature doesn't have the same importance. That is why we can proceed to a **hyper parameter tuning** depending on the features with the best predictions.

# In[42]:


predictors = [x for
              x in df.columns[0:len(df.columns)-1]]
print(predictors)


# This list of predictors is composed of the parameters that we have to tune according to their importance calculated below.

# In[43]:


feature_importance = pd.Series(gradient_boosting_model.feature_importances_,predictors).sort_values(ascending=False)
fig = plt.figure(figsize=(12,6))
print(feature_importance)
feature_importance.plot(kind = 'bar',title = 'Feature Importance')
plt.ylabel('Feature Importance Score')


# We try the model with the most important features. We tried many combinations taking from 1 to 27 predictors.

# In[44]:


x_new = df[['discharge_disposition_id','number_inpatient','number_diagnoses','age','number_emergency','number_outpatient','num_medications','num_lab_procedures','admission_type_id','diabetesMed','diag_1','time_in_hospital']]
y_new = df['readmitted']
x_train_new, x_test_new , y_train_new ,y_test_new = train_test_split(x_new,y_new,test_size=0.33)

#print("X train: ",x_train_new.shape)
#print("X test: ",x_test_new.shape)
#print("Y train: ",y_train_new.shape)
#print("Y test: ",y_test_new.shape)

scaler = StandardScaler()
scaler.fit(x_train_new)
x_train_new = scaler.transform(x_train_new)
x_test_new = scaler.transform(x_test_new)


# In[45]:


naive_Bayes_model_new = GaussianNB()
naive_Bayes_model_new.fit(x_train_new,y_train_new)
naive_Bayes_model_pred_new = naive_Bayes_model_new.predict(x_test_new)

random_forest_model_new = RandomForestClassifier()
random_forest_model_new.fit(x_train_new,y_train_new)
random_forest_model_pred_new = random_forest_model_new.predict(x_test_new)

gradient_boosting_model_new = GradientBoostingClassifier() 
gradient_boosting_model_new.fit(x_train_new,y_train_new)
gradient_boosting_model_pred_new = gradient_boosting_model_new.predict(x_test_new)


# In[46]:


naive_Bayes_model_accuracy_new = naive_Bayes_model_new.score(x_test_new,y_test_new)
random_forest_model_accuracy_new = random_forest_model_new.score(x_test_new,y_test_new)
gradient_boosting_model_accuracy_new = gradient_boosting_model_new.score(x_test_new,y_test_new)

results = pd.DataFrame({
    'Model':['Naive Bayes','Random Forest','Gradient Boosting'],
    'Score':[naive_Bayes_model_accuracy_new,random_forest_model_accuracy_new,gradient_boosting_model_accuracy_new]
})

results_df = results.sort_values(by='Score',ascending=False)
results_df = results_df.set_index('Score')
results_df


# In[47]:


x_val_train_new, x_val_test_new, y_val_train_new, y_val_test_new = train_test_split(x_test_new, y_test_new, test_size=0.5, random_state=101)

naive_Bayes_model_pred_new = naive_Bayes_model_new.predict(x_val_train_new)
random_forest_model_pred_new = random_forest_model_new.predict(x_val_train_new)
gradient_boosting_model_pred_new = gradient_boosting_model_new.predict(x_val_train_new)

print('\nPerformances:\n\n')
print('F1 Score Bayes       : ', metrics.f1_score(y_val_train_new, naive_Bayes_model_pred_new,average='micro'))
print('Confusion Matrix Bayes:\n ', confusion_matrix(y_val_train_new, naive_Bayes_model_pred_new))
print('\n')
print('F1 Score Random Forest       : ', metrics.f1_score(y_val_train_new, random_forest_model_pred_new,average='micro'))
print('Confusion Matrix Random Forest:\n ', confusion_matrix(y_val_train_new, random_forest_model_pred_new))
print('\n')
print('F1 Score Gradient       : ', metrics.f1_score(y_val_train_new, gradient_boosting_model_pred_new,average='micro'))
print('Confusion Matrix Gradient:\n ', confusion_matrix(y_val_train_new, gradient_boosting_model_pred_new))


# We noticed some facts:
#  - The less there are features, the more the **Bayes** model is efficient.
#  - The more there are features, the more the **Gradient Boosting** model is efficient.
# 
# After many attempts, we managed to obtain an accuracy about <ins>62.5%</ins>.

# ### Gradiant Boosting Model Tuning

# We have to choose the best number of parameters in order to improve the learning rate.
# Thanks to this, the combination of the subsample models create a more powerful new model and we repeat this process.
# 
# Before that, we have to initialize the **boosting parameters**:
# 
#  - **min_samples_split** : this should be a small proportion of our dataset (~1%)
#  - **min_samples_leaf** : this prevent from overfitting. This is based on testing and intuition.
#  - **max_depth** : this is based on the number of observation and predictors (5-8)
#  - **max_features** : usually start with square root
#  - **subsample** : usually 0.8
#  - **n_estimators** : this is the number of grid search to check the optimum number of trees. It tests out values from a range by a step that we choose.
# 
# Theses parameters are initialized. They after could be tuned in order to find the **optimum combination for the best accuracy score**.

# In[48]:


param_test = {'n_estimators':range(20,75,5)}
gsearch = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, 
                                                               min_samples_split=350, 
                                                               min_samples_leaf=50,
                                                               max_depth=8,
                                                               max_features='sqrt', 
                                                               subsample=0.8,
                                                               random_state=101), param_grid = param_test, scoring='f1_micro',n_jobs=-1,cv=5)
gsearch.fit(x_train,y_train)


# In[49]:


gsearch.best_params_, gsearch.best_score_


# We finally have a model based on **Gradient Boosting** wich is boosted by the **model tuning** thanks to a **grid search**. The best accucary that we obtained through this whole work is **62,5%**.
