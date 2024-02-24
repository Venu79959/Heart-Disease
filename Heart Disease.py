#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Analysis by KODI VENU

# In[2]:


import pandas as pd
import numpy as np
data = pd.read_csv('heart.csv')


# Displaying Top 5 Rows of the Dataset

# In[3]:


data.head(5)


# Displaying Last 5 Rows of the Dataset

# In[5]:


data.tail(5)


# Finding shape of the Dataset

# In[6]:


data.shape


# In[7]:


print("Number of rows", data.shape[0])
print("Number of columns", data.shape[1])


# Dataset Information

# In[8]:


data.info()


# Checking Null Values in the dataset

# In[10]:


data.isnull()


# In[11]:


data.isnull().sum()


# Checking for Duplicate Data & Drop them

# In[12]:


data_dup=data.duplicated().any()
print(data_dup)


# In[13]:


data=data.drop_duplicates()


# In[14]:


data.shape


# In[15]:


data_dup=data.duplicated().any()
print(data_dup)


# Dataset Overall Statistics

# In[9]:


data.describe()


# In[26]:


get_ipython().system('pip install matplotlib')


# Draw correlation matrix

# In[16]:


data.corr()


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(25,9))
sns.heatmap(data.corr(), annot=True)


# how many people have heart disease and how many dont have heart disease

# In[29]:


data.columns


# In[30]:


data['target'].value_counts()


# In[54]:


sns.countplot(data['target'])
plt.figure(figsize=(13,6))


# find count of male & female in this dataset

# In[33]:


data.columns


# In[35]:


data['sex'].value_counts()


# In[44]:


sns.countplot(data['sex'])
plt.xticks([0,1],['Female','Male'])
plt.show()


# find gender distribution according to the target variable

# In[42]:


data.columns


# In[48]:


sns.countplot(x='sex',hue='target',data=data)
plt.xticks([1,0],['Male','Female'])
plt.legend(labels=['No-Disease','Disease'])
plt.show()


# check age distribution in the dataset

# In[50]:


sns.distplot(data['age'],bins=20)
plt.show()


# check chest pain type

# In[56]:


sns.countplot(data['cp'])
plt.xticks([0,1,2,3],["typical angina","atypical angina","non-anginal pain","asymptomatic"])
plt.xticks(rotation=75)
plt.show()


# show the chest pain distribution as per target variable

# In[57]:


data.columns


# In[59]:


sns.countplot(x='cp',hue="target",data=data)
plt.legend(labels=['No-Disease','Disease'])
plt.show()


# show fasting blood sugar distribution according to target variable

# In[60]:


sns.countplot(x='fbs',hue="target",data=data)
plt.legend(labels=['No-Disease','Disease'])
plt.show()


# check resting blood pressure distribution 

# In[61]:


data.columns


# In[62]:


data['trestbps'].hist()


# compare resting blood pressure as per sex column

# In[65]:


g=sns.FacetGrid(data,hue='sex', aspect=4)
g.map(sns.kdeplot,'trestbps',shade=True)
plt.legend(labels=['Male','Female'])
plt.show()


# show distribution of serum cholestrol

# In[66]:


data.columns


# In[67]:


data['chol'].hist()


# Plot continuous variables

# In[68]:


data.columns


# In[71]:


cate_val=[]
cont_val=[]

for column in data.columns:
    if data[column].nunique() <= 10:
        cate_val.append(column)
    else:
        cont_val.append(column)


# In[72]:


cate_val


# In[73]:


cont_val


# In[75]:


data.hist(cont_val, figsize=(15,6))
plt.tight_layout()
plt.show()


# In[ ]:




