#!/usr/bin/env python
# coding: utf-8

# # Foundations of Data Science Project - Diabetes Analysis
# 
# ---------------
# ## Context
# ---------------
# 
# Diabetes is one of the most frequent diseases worldwide and the number of diabetic patients are growing over the years. The main cause of diabetes remains unknown, yet scientists believe that both genetic factors and environmental lifestyle play a major role in diabetes.
# 
# A few years ago research was done on a tribe in America which is called the Pima tribe (also known as the Pima Indians). In this tribe, it was found that the ladies are prone to diabetes very early. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients were females at least 21 years old of Pima Indian heritage. 
# 
# -----------------
# ## Objective
# -----------------
# 
# Here, we are analyzing different aspects of Diabetes in the Pima Indians tribe by doing Exploratory Data Analysis.
# 
# -------------------------
# ## Data Dictionary
# -------------------------
# 
# The dataset has the following information:
# 
# * Pregnancies: Number of times pregnant
# * Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
# * BloodPressure: Diastolic blood pressure (mm Hg)
# * SkinThickness: Triceps skin fold thickness (mm)
# * Insulin: 2-Hour serum insulin (mu U/ml)
# * BMI: Body mass index (weight in kg/(height in m)^2)
# * DiabetesPedigreeFunction: A function which scores likelihood of diabetes based on family history.
# * Age: Age in years
# * Outcome : Class variable (0: person is not diabetic or 1: person is diabetic)

# ## Q 1: Import the necessary libraries and briefly explain the use of each library (3 Marks)

# In[1]:


#import appropriate library

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Write your Answer here: 
numpy: is for working with various numerical data in python.

pandas: used for all steps (cleaning, sorting, manipulating) for preparing data befpre start analyzing.

matplotlib.pyplot: a visualization library from the Matlab software and .pyplot is related to python to help to visualize data. 

seaborn:a visualization library which is needed for working with statistical models like heat maps.
# ## Q 2: Read the given dataset (1 Mark)

# In[23]:


#read the file

pima = pd.read_csv("diabetes.csv")


# ## Q3. Show the last 10 records of the dataset. How many columns are there? (1 Mark)

# In[24]:


#last 10 rows
pima.tail(10)


# #### Write your Answer here: 
# 
Ans 3: 9 columns
# ## Q4. Show the first 10 records of the dataset (1 Mark)

# In[25]:


#first 10 rows

pima.head(10)


# ## Q5. What do you understand by the dimension of the dataset? Find the dimension of the `pima` dataframe. (1 Mark)

# In[26]:


#shape and dimention

pima.shape


# #### Write your Answer here: 
# 
Ans 5: with these 2 eleemnts in the tuple we know there are 768 rows(include headers) and 9 columns in the dataset.
# ## Q6. What do you understand by the size of the dataset? Find the size of the `pima` dataframe. (1 Mark)

# In[27]:


#size of the dataset
pima.size


# #### Write your Answer here: 
# 
Ans 6: there are 6912 elements in total which comes from 768*9
# ## Q7. What are the data types of all the variables in the data set? (2 Marks)
# **Hint: Use info() function to get all the information about the dataset.**

# In[28]:


#check the data type
pima.info()


# #### Write your Answer here: 
# 
Ans 7: there are 2 different types of data, both numerical, some integers and some floats.
# ## Q8. What do you mean by missing values? Are there any missing values in the `pima` dataframe? (2 Marks)

# In[29]:


#if there is any missing value

pima.isnull().values.any()


# #### Write your Answer here: 
# 
Ans 8: any cell in dataset which is blank and empty is consider as missing values, which here we dont have any and that's why we have False as a result.
# ## Q9. What does summary statistics of data represents? Find the summary statistics for all variables except 'Outcome' in the `pima` data? Take one column/variable from the output table and explain all the statistical measures. (3 Marks)

# In[30]:


#statistic information
pima.iloc[:,0:8].describe()


# In[31]:


pima.iloc[:,0:1].describe()


# #### Write your Answer here: 
# 
Ans 9: total number of pregnancies is 768 times, between none/0 as minimum and 17 times as maximum.
the average nuber of pregnancies is ~4 times. standard deviation of this given set of numbers is ~3.5 , in the first 25% of data number of pregnancies is 1 time and in the 50% of data number of pregnancies is 3 times which is the median and in the last 25% of data (after 75%) number of pregnancies is 6 times.
# ## Q 10. Plot the distribution plot for the variable 'BloodPressure'. Write detailed observations from the plot. (2 Marks)

# In[32]:


#distribution 

sns.displot(pima['BloodPressure'], kind='kde')
plt.show()


# #### Write your Answer here: 
# 
Ans 10: The plot shows that most of the observations lies between 60 and 80.
The distribution doesn't look symmetric and there are two total separate peaks in the plot around 0 and 70. It is a multimodal.
# ## Q 11. What is the 'BMI' for the person having the highest 'Glucose'? (1 Mark)

# In[33]:


#get the maximum Gloucose

pima[pima['Glucose']==pima['Glucose'].max()]['BMI']


# #### Write your Answer here: 
# 
Ans 11: it is 42.9
# ## Q12.
# ### 12.1 What is the mean of the variable 'BMI'? 
# ### 12.2 What is the median of the variable 'BMI'? 
# ### 12.3 What is the mode of the variable 'BMI'?
# ### 12.4 Are the three measures of central tendency equal?
# 
# ### (3 Marks)

# In[47]:


#find mean, median and mode

m1 = pima['BMI'].mean()  #Mean
print(m1)

m2 = pima['BMI'].median()
print(m2)

m3 = pima['BMI'].mode()[0]  #Mode
print(m3)


# #### Write your Answer here: 
# 
Ans 12: They are almost equal.
# ## Q13. How many women's 'Glucose' level is above the mean level of 'Glucose'? (1 Mark)

# In[48]:


#Glucose level higher than mean

pima[pima['Glucose']>pima['Glucose'].mean()].shape[0]


# #### Write your Answer here: 
# 
Ans 13: 349 women has a higher level og Gloucose than the mean level
# ## Q14. How many women have their 'BloodPressure' equal to the median of 'BloodPressure' and their 'BMI' less than the median of 'BMI'? (2 Marks)

# In[52]:


#BloodPressure equal to the median
#BMI less than the median 

pima[(pima['BloodPressure']==pima['BloodPressure'].median()) & (pima['BMI']<pima['BMI'].median())] 


# #### Write your Answer here: 
# 
Ans 14: 22 women have their 'BloodPressure' equal to the median of 'BloodPressure' and their 'BMI' less than the median of 'BMI'.
# ## Q15. Create the pairplot for variables 'Glucose', 'SkinThickness' and 'DiabetesPedigreeFunction'. Write you observations from the plot. (4 Marks)

# In[53]:


#pairplot

sns.pairplot(data=pima,vars=['Glucose', 'SkinThickness', 'DiabetesPedigreeFunction'], hue='Outcome')
plt.show()


# #### Write your Answer here: 
# 
Ans 15: The histogram on the diagonal allows us to see the distribution of a single variable while the scatter plots on the upper and lower triangles show the relationship between two variables (which is not easy to find a accure positive or negative correlation between valus.) 
# ## Q16. Plot the scatterplot between 'Glucose' and 'Insulin'. Write your observations from the plot. (2 Marks)

# In[55]:


#scatterplot

sns.scatterplot(x='Glucose',y='Insulin',data=pima)
plt.show()


# #### Write your Answer here: 
# 
Ans 16: I beleive we can not say there is strong correlation between variables because the data are not quite a round og best fit line, yet it seems there is a relation between Glucose level and Insulin but it's not accurate to predict or train a model.
# ## Q 17. Plot the boxplot for the 'Age' variable. Are there outliers? (2 Marks)

# In[56]:


#boxplot

plt.boxplot(pima['Age'])

plt.title('Boxplot of Age')
plt.ylabel('Age')
plt.show()


# #### Write your Answer here: 
# 
Ans 17: yes there are some outliers and one is quite far from the rest.
# ## Q18. Plot histograms for variable Age to understand the number of women in different Age groups given that they have diabetes or not. Explain both histograms and compare them. (3 Marks)

# In[58]:


#histogram

plt.hist(pima[pima['Outcome']==1]['Age'], bins = 5)
plt.title('Distribution of Age for Women who has Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[59]:


#histogram

plt.hist(pima[pima['Outcome']==0]['Age'], bins = 5)
plt.title('Distribution of Age for Women who do not have Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# #### Write your Answer here: 
# 
Ans 18: in both plots the number of women with diabetes and without is higher in younger ages, and it seems after 30 years the risk of having diabetes increasing, that's why in second plot the bin betwen 35-45 is dropping deramatically.
# ## Q 19. What is Inter Quartile Range of all the variables? Why is it used? Which plot visualizes the same? (2 Marks)

# In[60]:


#inter quartile

Q1 = pima.quantile(0.25)
Q3 = pima.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# #### Write your Answer here: 
# 
Ans 19: The IQR is used to measure how spread out the data points in a set are from the mean of the data set. The higher the IQR, the more spread out the data points; in contrast, the smaller the IQR, the more bunched up the data points are around the mean. 
with box plot we can see this distribution of data and also our outliers.
# ## Q 20. Find and visualize the the correlation matrix. Write your observations from the plot. (3 Marks)

# In[61]:


#correlation matrix

corr_matrix = pima.iloc[:,0:8].corr()

corr_matrix


# In[62]:


#heatmap

plt.figure(figsize=(8,8))
sns.heatmap(corr_matrix, annot = True)

# display the plot
plt.show()


# #### Write your Answer here: 
# 
Ans 20: there is neither strong positive nor strong negative correlation between numeric variables. there is only a kind of relation between age and pregnancies which is not strong.