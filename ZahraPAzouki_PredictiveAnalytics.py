#!/usr/bin/env python
# coding: utf-8

# **New York City Taxi Ride Duration Prediction**
# 
# In this case study, we will build a predictive model to predict the duration of BUS ride. We will do the following steps:
#   * Install the dependencies
#   * Load the data as pandas dataframe
#   * Define the outcome variable - the variable we are trying to predict.
#   * Build features with Deep Feature Synthesis using the [featuretools](https://featuretools.com) package. We will start with simple features and incrementally improve the feature definitions and examine the accuracy of the system

# In[1]:


get_ipython().system('pip install featuretools==0.27.0')


# #### Note: If !pip install featuretools doesn't work, please install using the anaconda prompt by typing the following command in anaconda prompt
#       conda install -c conda-forge featuretools==0.27.0

# **Install Dependencies**
# <p>If you have not done so already, download this repository <a href="https://github.com/Featuretools/DSx/archive/master.zip">from git</a>. Once you have downloaded this archive, unzip it and cd into the directory from the command line. Next run the command ``./install_osx.sh`` if you are on a mac or ``./install_linux.sh`` if you are on Linux. This should install all of the dependencies.</p>
# <p> If you are on a windows machine, open the requirements.txt folder and make sure to install each of the dependencies listed (featuretools, jupyter, pandas, sklearn, numpy) </p>
# <p> Once you have installed all of the dependencies, open this notebook. On Mac and Linux, navigate to the directory that you downloaded from git and run ``jupyter notebook`` to be taken to this notebook in your default web browser. When you open the NewYorkCity_taxi_case_study.ipynb file in the web browser, you can step through the code by clicking the ``Run`` button at the top of the page. If you have any questions for how to use <a href="http://jupyter.org/">Jupyter</a>, refer to Google or the discussion forum.</p>

# ### Importing libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Feataurestools for feature engineering
import featuretools as ft

#Utils python file contains some fuctions which can be used any where you want. 
import utils
from utils import load_nyc_taxi_data, compute_features, preview, feature_importances

# Importing gradient boosting regressor, to make prediction
from sklearn.ensemble import GradientBoostingRegressor

#importing primitives
from featuretools.primitives import (Minute, Hour, Day, Week, Month,
                                     Weekday, IsWeekend, Count, Sum, Mean, Median, Std, Min, Max)

print(ft.__version__)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Step 1: Load the raw data

# In[3]:


trips, pickup_neighborhoods, dropoff_neighborhoods = load_nyc_taxi_data()
preview(trips, 10)


# The ``trips`` table has the following fields
# * ``id`` which uniquely identifies the trip
# * ``vendor_id`` is the taxi cab company - in our case study we have data from three different cab companies
# * ``pickup_datetime`` the time stamp for pickup
# * ``dropoff_datetime`` the time stamp for drop-off
# * ``passenger_count`` the number of passengers for the trip
# * ``trip_distance`` total distance of the trip in miles 
# * ``pickup_longitude`` the longitude for pickup
# * ``pickup_latitude`` the latitude for pickup
# * ``dropoff_longitude``the longitude of dropoff 
# * ``dropoff_latitude`` the latitude of dropoff
# * ``payment_type`` a numeric code signifying how the passenger paid for the trip. 1= Credit card 2= Cash 3= No charge 4= Dispute 5= Unknown 6= Voided
# * ``trip_duration`` this is the duration we would like to predict using other fields 
# * ``pickup_neighborhood`` a one or two letter id of the neighborhood where the trip started
# * ``dropoff_neighborhood`` a one or two letter id of the neighborhood where the trip ended

# ### Step 2: Prepare the Data
# 
# Lets create entities and relationships. The three entities in this data are 
# * trips 
# * pickup_neighborhoods
# * dropoff_neighborhoods
# 
# This data has the following relationships
# * pickup_neighborhoods --> trips (one neighborhood can have multiple trips that start in it. This means pickup_neighborhoods is the ``parent_entity`` and trips is the child entity)
# * dropoff_neighborhoods --> trips (one neighborhood can have multiple trips that end in it. This means dropoff_neighborhoods is the ``parent_entity`` and trips is the child entity)
# 
# In <a <href="https://www.featuretools.com/"><featuretools (automated feature engineering software package)/></a>, we specify the list of entities and relationships as follows: 
# 

# ### Question 1: Define entities and relationships for the Deep Feature Synthesis (5 Marks)

# In[4]:


entities = {
        "trips": (trips, "id", 'pickup_datetime' ),
        "pickup_neighborhoods": (pickup_neighborhoods, "neighborhood_id"),
        "dropoff_neighborhoods": (dropoff_neighborhoods, "neighborhood_id"),
        }

relationships = [("pickup_neighborhoods", "neighborhood_id", "trips", "pickup_neighborhood"),
                 ("dropoff_neighborhoods", "neighborhood_id", "trips", "dropoff_neighborhood")]


# Next, we specify the cutoff time for each instance of the target_entity, in this case ``trips``.This timestamp represents the last time data can be used for calculating features by DFS. In this scenario, that would be the pickup time because we would like to make the duration prediction using data before the trip starts. 
# 
# For the purposes of the case study, we choose to only select trips that started after January 12th, 2016. 

# In[5]:


cutoff_time = trips[['id', 'pickup_datetime']]
cutoff_time = cutoff_time[cutoff_time['pickup_datetime'] > "2016-01-12"]
preview(cutoff_time, 10)


# ### Step 3: Create baseline features using Deep Feature Synthesis
# 
# Instead of manually creating features, such as "month of pickup datetime", we can let DFS come up with them automatically. It does this by 
# * interpreting the variable types of the columns e.g categorical, numeric and others 
# * matching the columns to the primitives that can be applied to their variable types
# * creating features based on these matches

# **Create transform features using transform primitives**
# 
# As we described in the video, features fall into two major categories, ``transform`` and ``aggregate``. In featureools, we can create transform features by specifying ``transform`` primitives. Below we specify a ``transform`` primitive called ``weekend`` and here is what it does:
# 
# * It can be applied to any ``datetime`` column in the data. 
# * For each entry in the column, it assess if it is a ``weekend`` and returns a boolean. 
# 
# In this specific data, there are two ``datetime`` columns ``pickup_datetime`` and ``dropoff_datetime``. The tool automatically creates features using the primitive and these two columns as shown below. 

# ### Question 2: Create a model with only 1 transform primitive (10 Marks)
# 
# **Question: 2.1 Define transform primitive for weekend and define features using dfs?**

# In[6]:


#defining Transform feature as weather the ride was at the weekend or not, and it is affecting the duration of the ride. 
trans_primitives = [IsWeekend]

#defining features we want to create using featuretools deep featurs synthesis(dfs)
features = ft.dfs(entities=entities,
                  relationships=relationships,
                  target_entity="trips",
                  trans_primitives=trans_primitives,
                  agg_primitives=[],
                  ignore_variables={"trips": ["pickup_latitude", "pickup_longitude",
                                              "dropoff_latitude", "dropoff_longitude"]},
                  features_only=True)


# *If you're interested about parameters to DFS such as `ignore_variables`, you can learn more about these parameters [here](https://docs.featuretools.com/generated/featuretools.dfs.html#featuretools.dfs)*
# <p>Here are the features created.</p>

# In[7]:


print ("Number of features: %d" % len(features))
features


# Now let's compute the features. 

# **Question: 2.2 Compute features and define feature matrix**

# In[8]:


def compute_features(features, cutoff_time):
    # shuffle so we don't see encoded features in the front or backs

    np.random.shuffle(features)
    feature_matrix = ft.calculate_feature_matrix(features,
                                                 cutoff_time=cutoff_time,
                                                 approximate='36d',
                                                 verbose=True,entities=entities, relationships=relationships)
    print("Finishing computing...")
    feature_matrix, features = ft.encode_features(feature_matrix, features,
                                                  to_encode=["pickup_neighborhood", "dropoff_neighborhood"],
                                                  include_unknown=False)
    return feature_matrix


# In[9]:


feature_matrix1 = compute_features(features, cutoff_time)


# In[10]:


preview(feature_matrix1, 5)


# ### Build the Model
# 
# To build a model, we
# * Separate the data into a portion for ``training`` (75% in this case) and a portion for ``testing`` 
# * Get the log of the trip duration so that a more linear relationship can be found.
# * Train a model using a ``GradientBoostingRegressor``
# 
# **Question: 2.3 What was the Modeling Score after your last training round?**
# 
# **Question: 2.4 Hypothesize on how including more robust features will change the accuracy.**

# In[11]:


# separates the whole feature matrix into train data feature matrix, 
# train data labels, and test data feature matrix 
X_train1, y_train1, X_test1, y_test1 = utils.get_train_test_fm(feature_matrix1,.75)
y_train1 = np.log(y_train1 +1)
y_test1 = np.log(y_test1 +1)


# In[12]:


model1 = GradientBoostingRegressor(verbose=True)
model1.fit(X_train1, y_train1)
model1.score(X_test1, y_test1)


# **The score for the model with transform primitive is ~72.2% . The more robust features we add incrementally, we could see that the accuracy improves by observing the increasing modeling score.**

# ### Step 5: Adding more Transform Primitives
# 
# * Add ``Minute``, ``Hour``, ``Week``, ``Month``, ``Weekday`` , etc primitives
# * All these transform primitives apply to ``datetime`` columns
# 
# ### Question 3: Create a model with more transform primitives (10 Marks)
# 
# **3.1 Define more transform primitives and define features using dfs?**

# In[13]:


trans_primitives = [Minute, Hour, Day, Week, Month, Weekday, IsWeekend]

features = ft.dfs(entities=entities,
                  relationships=relationships,
                  target_entity="trips",
                  trans_primitives=trans_primitives,
                  agg_primitives=[],
                  ignore_variables={"trips": ["pickup_latitude", "pickup_longitude",
                                              "dropoff_latitude", "dropoff_longitude"]},
                  features_only=True)


# In[14]:


print ("Number of features: %d" % len(features))
features


# Now let's compute the features. 

# **Question: 3.2 Compute features and define feature matrix**

# In[15]:


feature_matrix2 = compute_features(features, cutoff_time)


# In[16]:


preview(feature_matrix2, 10)


# ### Step 6: Build the new model
# 
# **Question: 3.3 What was the Modeling Score after your last training round when including the transform primitives?**
# 
# **Question: 3.4 Comment on how the modeling accuracy differs when including more transform features.**

# In[17]:


# separates the whole feature matrix into train data feature matrix,
# train data labels, and test data feature matrix 
X_train2, y_train2, X_test2, y_test2 = utils.get_train_test_fm(feature_matrix2,.75)
y_train2 = np.log(y_train2+1)
y_test2 = np.log(y_test2+1)


# In[18]:


#remove ________ and write the code
model2 = GradientBoostingRegressor(verbose=True)
model2.fit(X_train2,y_train2)
model2.score(X_test2,y_test2)


# **As we can see the score for the model is better than previous one, and improved from ~72.2% to ~77.5% , and the time for computaito is not that long. we will check the model performance with aggregate primitive as well**

# ### Step 7: Add Aggregation Primitives
# 
# Now let's add aggregation primitives. These primitives will generate features for the parent entities ``pickup_neighborhoods``, and ``dropoff_neighborhood`` and then add them to the trips entity, which is the entity for which we are trying to make prediction.

# ### Question 4: Create a model with transform and aggregate primitive (10 Marks)
# **4.1 Define more transform and aggregate primitive and define features using dfs?**

# In[19]:


trans_primitives = [Minute, Hour, Day, Week, Month, Weekday, IsWeekend]
aggregation_primitives = [Count, Sum, Mean, Median, Std, Max, Min]

features = ft.dfs(entities=entities,
                  relationships=relationships,
                  target_entity="trips",
                  trans_primitives=trans_primitives,
                  agg_primitives=aggregation_primitives,
                  ignore_variables={"trips": ["pickup_latitude", "pickup_longitude",
                                              "dropoff_latitude", "dropoff_longitude"]},
                  features_only=True)


# In[20]:


print ("Number of features: %d" % len(features))
features


# **Question: 4.2 Compute features and define feature matrix**

# In[21]:


feature_matrix3 = compute_features(features, cutoff_time)


# In[22]:


preview(feature_matrix3, 10)


# ### Step 8: Build the new model
# 
# **Question 4.3 What was the Modeling Score after your last training round when including the aggregate transforms?**
# 
# **Question 4.4 How do these aggregate transforms impact performance? How do they impact training time?**

# In[23]:


# separates the whole feature matrix into train data feature matrix,
# train data labels, and test data feature matrix 
X_train3, y_train3, X_test3, y_test3 = utils.get_train_test_fm(feature_matrix3,.75)
y_train3 = np.log(y_train3 + 1)
y_test3 = np.log(y_test3 + 1)


# In[24]:


# note: this may take up to 20 minutes to run
model3 = GradientBoostingRegressor(verbose=True)
model3.fit(X_train3, y_train3)

model3.score(X_test3,y_test3)


# **The computation time was so long, yet the result is not that better than model 2,The score for the model with more aggregate transforms is ~77.8% and has improved only by a small amount. so it seems considering training time the best model is model 2 by using transform features.**

# #### Based on the above 3 models, we can make predictions using our model2, as it is giving almost same accuracy as model3 and also the training time is not that large as compared to model3

# In[25]:


y_pred = model2.predict(X_test2)
y_pred = np.exp(y_pred) - 1 # undo the log we took earlier
y_pred[5:]


# ### Question 5: What are some important features based on model2 and how can they affect the duration of the rides? (5 Marks)

# In[26]:


feature_importances(model2, feature_matrix2.columns, n=15)


# **IS_WEEKEND(dropoff_datetime) is the most important feature, which implies that during weekend the rides are longer, specially dropoff (people usually don't drive after parties, bars, etc and take taxi).
# Other important features is hour, it seems depending on the time (rush hour, late, etc) the duration of the rides might be longer.**
