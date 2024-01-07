#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('mtcars.csv')


# In[3]:


df.head()


# In[6]:


#bar chart
x = df['mpg']
y = df['qsec']
plt.bar(x,y, color ='orange')


# In[7]:


#piechart
y = df['model']
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(x, autopct='%.1f%%')
ax.set_title('Model')
plt.tight_layout()


# In[14]:


#histogram
plt.hist(df['mpg'],bins=[0,10,20,30,40])


# In[16]:


#boxplot
new_data = df[["qsec"]]
print(new_data.head())
plt.figure(figsize = (9, 7))
new_data.boxplot()


# In[15]:


plt.scatter(df['mpg'],df['hp'],color='red')
plt.xlabel('mileage')
plt.ylabel('horsepower')
plt.title('mpg vs hp')


# In[17]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:




