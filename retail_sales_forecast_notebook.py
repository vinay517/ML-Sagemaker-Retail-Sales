#!/usr/bin/env python
# coding: utf-8

# # TASK #1: UNDERSTAND THE PROBLEM STATEMENT/GOAL
# 
# 

# - This dataset contains weekly sales from 99 departments belonging to 45 different stores. 
# - Our aim is to forecast weekly sales from a particular department.
# - The objective of this case study is to forecast weekly retail store sales based on historical data.
# - The data contains holidays and promotional markdowns offered by various stores and several departments throughout the year.
# - Markdowns are crucial to promote sales especially before key events such as Super Bowl, Christmas and Thanksgiving. 
# - Developing accurate model will enable make informed decisions and make recommendations to improve business processes in the future. 
# - The data consists of three sheets: 
#     - Stores
#     - Features
#     - Sales
# - Data Source : https://www.kaggle.com/manjeetsingh/retaildataset

# # TASK #2: IMPORT DATASET AND LIBRARIES

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile


# In[4]:


# import the csv files using pandas 
feature = pd.read_csv('Features_data_set.csv')
sales = pd.read_csv('sales_data_set.csv')
stores = pd.read_csv('stores_data_set.csv')


# In[5]:


# Let's explore the 3 dataframes
# "stores" dataframe contains information related to the 45 stores such as type and size of store.

stores


# In[6]:


# Let's explore the "feature" dataframe
# Features dataframe contains additional data related to the store, department, and regional activity for the given dates.
# Store: store number
# Date: week
# Temperature: average temperature in the region
# Fuel_Price: cost of fuel in the region
# MarkDown1-5: anonymized data related to promotional markdowns. 
# CPI: consumer price index
# Unemployment: unemployment rate
# IsHoliday: whether the week is a special holiday week or not

feature


# In[7]:


# Let's explore the "sales" dataframe
# "Sales" dataframe contains historical sales data, which covers 2010-02-05 to 2012-11-01. 
# Store: store number
# Dept: department number
# Date: the week
# Weekly_Sales: sales for the given department in the given store
# IsHoliday: whether the week is a special holiday week

sales


# # TASK #3: EXPLORE INDIVIDUAL DATASET

# In[8]:


# max fuel price is 4.468
# max unemployment numbers 14.313000
# avg size of stores 130287.60


# MINI CHALLENGE
# - Use info and describe to individually explore the 3 dataframes
# - What is the maximum fuel price? and maximum unemployment numbers?
# - What is the average size of the stores?
# 

# In[9]:


sales.info()


# In[10]:


feature.info()


# In[11]:


stores.info()


# In[12]:


sales.describe()


# In[13]:


stores.describe()


# In[14]:


feature.describe()


# In[15]:


sns.heatmap(sales.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[16]:


sns.heatmap(feature.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[17]:


sns.heatmap(stores.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[18]:


sales.isnull().sum()


# In[19]:


stores.isnull().sum()


# In[20]:


feature.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


# Change the datatype of 'date' column

feature['Date'] = pd.to_datetime(feature['Date'])
sales['Date'] = pd.to_datetime(sales['Date'])


# In[22]:


feature


# In[23]:


sales


# # TASK #4: MERGE DATASET INTO ONE DATAFRAME

# In[24]:


sales.head()


# In[25]:


feature.head()


# In[26]:


df = pd.merge(sales, feature, on = ['Store','Date','IsHoliday'])


# In[27]:


df


# In[28]:


df.head()


# In[29]:


stores.head()


# In[30]:


df = pd.merge(df, stores, on = ['Store'], how = 'left')


# In[31]:


df.head()


# In[32]:


x = '2010-05-02'
str(x).split('-')


# MINI CHALLENGE
# - Define a function to extract the month information from the dataframe column "Date"
# - Apply the function to the entire column "Date" in the merged dataframe "df" and write the output in a column entitled "month"
# 

# In[33]:


# def get_month(x):
#     return int(str(x).split('-')[1])


# In[34]:


# df['month'] = df['Date'].apply(get_month)


# In[35]:


df['month'] = pd.DatetimeIndex(df['Date']).month
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # TASK #5: EXPLORE MERGED DATASET

# In[36]:


sns.heatmap(df.isnull(), cbar = False)


# In[37]:


# check the number of non-null values in the dataframe
df.isnull().sum()


# In[38]:


# Fill up NaN elements with zeros
df = df.fillna(0)


# In[39]:


df


# In[40]:


# Statistical summary of the combined dataframe
df.describe()


# In[41]:


# check the number of duplicated entries in the dataframe
df.duplicated().sum()


# In[42]:


df['Type'].value_counts()


# MINI CHALLENGE
# - Replace the "IsHoliday" with ones and zeros instead of True and False (characters with numbers)
# 

# In[43]:


# df['IsHoliday'] = df['IsHoliday'].apply(lambda x: 1 if x == 'False' else 0)
df['IsHoliday'] = df['IsHoliday'].replace({True:1, False:0})


# In[44]:


df


# # TASK #6: PERFORM EXPLORATORY DATA ANALYSIS

# In[45]:


# Create pivot tables to understand the relationship in the data

result = pd.pivot_table(df, values = 'Weekly_Sales', columns = ['Type'], index = ['Date', 'Store', 'Dept'],
                    aggfunc= np.mean)


# In[46]:


result


# In[47]:


result.describe()
# It can be seen that Type A stores have much higher sales than Type B and Type C


# In[48]:



result_md = pd.pivot_table(df, values = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'], columns = ['IsHoliday'], index = ['Date', 'Store','Dept'],
                    aggfunc={'MarkDown1' : np.mean,'MarkDown2' : np.mean, 'MarkDown3' : np.mean, 'MarkDown4' : np.mean, 'MarkDown5' : np.mean})


# In[49]:


result_md


# In[50]:


result_md.sum()


# In[51]:


result_md.describe()
# we can conclude that MarkDown2 and MarkDown3 have higher volume on holidays compared to that of regular days 
# while other MarkDowns don't show significant changes relating to holiday.


# In[52]:


corr_matrix = df.drop(columns = ['Store']).corr()


# In[53]:


plt.figure(figsize = (16,16))
sns.heatmap(corr_matrix, annot = True)
plt.show()


# # TASK #7: PERFORM DATA VISUALIZATION

# In[54]:


df


# In[55]:


df.hist(bins = 30, figsize = (20,20), color = 'r')


# In[56]:


# visualizing the relationship using pairplots
# there is a relationship between markdown #1 and Markdown #4
# holiday and sales 
# Weekly sales and markdown #3
sns.pairplot(df[["Weekly_Sales","IsHoliday","MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","Type","month"]], diag_kind = "kde")


# In[ ]:


df_type = df.groupby('Type').mean()


# In[ ]:


df_type


# In[57]:


sns.barplot(x = df['Type'], y = df['Weekly_Sales'], data = df)


# In[58]:


# df_dept = df.drop(columns = ['Store','Type','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment','Size','month'])
df_dept = df.groupby('Dept').mean()
df_dept


# In[59]:


fig = plt.figure(figsize = (14,16))
df_dept['Weekly_Sales'].plot(kind = 'barh', color = 'r', width = 0.9)


# In[60]:


fig = plt.figure(figsize = (14,16))
df_dept['MarkDown1'].plot(kind = 'barh', color = 'blue', width = 0.9)


# In[61]:


fig = plt.figure(figsize = (14,16))

df_dept['MarkDown2'].plot(kind = 'barh', color = 'yellow', width = 0.9)


# In[62]:


fig = plt.figure(figsize = (14,16))

df_dept['MarkDown3'].plot(kind = 'barh', color = 'black', width = 0.9)


# In[63]:


fig = plt.figure(figsize = (14,16))

df_dept['MarkDown4'].plot(kind = 'barh', color = 'orange', width = 0.9)


# In[64]:


fig = plt.figure(figsize = (14,16))

df_dept['MarkDown5'].plot(kind = 'barh', color = 'brown', width = 0.9)


# - We can conclude that departments that have poor weekly sales have been assigned high number of markdowns. Let's explore this in more details
# - Example: check out store 77 and 99 

# In[65]:


# Sort by weekly sales
df_dept_sale = df_dept.sort_values(by = ['Weekly_Sales'], ascending = True)
df_dept_sale['Weekly_Sales'][:30]


# # TASK #8: PREPARE THE DATA BEFORE TRAINING

# In[66]:


# Drop the date
df_target = df['Weekly_Sales']
df_final = df.drop(columns = ['Weekly_Sales', 'Date'])


# In[67]:


df_final = pd.get_dummies(df_final, columns = ['Type', 'Store', 'Dept'], drop_first = True)


# In[68]:


df_final.shape


# In[69]:


df_target.shape


# In[70]:


df_final


# In[71]:


X = np.array(df_final).astype('float32')
y = np.array(df_target).astype('float32')


# In[72]:


# reshaping the array from (421570,) to (421570, 1)
y = y.reshape(-1,1)
y.shape


# In[73]:


# scaling the data before feeding the model
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scaler_x = StandardScaler()
# X = scaler_x.fit_transform(X)

# scaler_y = StandardScaler()
# y = scaler_y.fit_transform(y)


# In[74]:


# spliting the data in to test and train sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5)


# In[75]:


X_train


# # TASK #9: TRAIN XGBOOST REGRESSOR IN LOCAL MODE

# In[76]:


get_ipython().system('pip install xgboost')


# In[77]:


# Train an XGBoost regressor model 

import xgboost as xgb


model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 5, n_estimators = 100)

model.fit(X_train, y_train)


# In[78]:


# predict the score of the trained model using the testing dataset

result = model.score(X_test, y_test)

print("Accuracy : {}".format(result))


# In[79]:


# make predictions on the test data

y_predict = model.predict(X_test)


# In[80]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 


# MINI CHALLENGE
# - Retrain the model with less 'max_depth'
# - Comment on the results

# In[ ]:





# # TASK #10: TRAIN XGBOOST USING SAGEMAKER

# In[55]:


# Convert the array into dataframe in a way that target variable is set as the first column and followed by feature columns
# This is because sagemaker built-in algorithm expects the data in this format.

train_data = pd.DataFrame({'Target': y_train[:,0]})
for i in range(X_train.shape[1]):
    train_data[i] = X_train[:,i]


# In[56]:


train_data.head()


# In[57]:


val_data = pd.DataFrame({'Target':y_val[:,0]})
for i in range(X_val.shape[1]):
    val_data[i] = X_val[:,i]


# In[58]:


val_data.head()


# In[59]:


val_data.shape


# In[60]:


# save train_data and validation_data as csv files.

train_data.to_csv('train.csv', header = False, index = False)
val_data.to_csv('validation.csv', header = False, index = False)


# In[73]:


# Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python
# Boto3 allows Python developer to write software that makes use of services like Amazon S3 and Amazon EC2

import sagemaker
import boto3

# Create a sagemaker session
sagemaker_session = sagemaker.Session()

#S 3 bucket and prefix that we want to use
# default_bucket - creates a Amazon S3 bucket to be used in this session
bucket = 'sagemaker-practical-3'
prefix = 'XGBoost-Regressor'
key = 'XGBoost-Regressor'
#Roles give learning and hosting access to the data
#This is specified while opening the sagemakers instance in "Create an IAM role"
role = sagemaker.get_execution_role()


# In[74]:


print(role)


# In[75]:


# read the data from csv file and then upload the data to s3 bucket
import os
with open('train.csv','rb') as f:
    # The following code uploads the data into S3 bucket to be accessed later for training
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(f)

# Let's print out the training data location in s3
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))


# In[76]:


# read the data from csv file and then upload the data to s3 bucket

with open('validation.csv','rb') as f:
    # The following code uploads the data into S3 bucket to be accessed later for training

    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', key)).upload_fileobj(f)
# Let's print out the validation data location in s3
s3_validation_data = 's3://{}/{}/validation/{}'.format(bucket, prefix, key)
print('uploaded validation data location: {}'.format(s3_validation_data))


# In[77]:


# creates output placeholder in S3 bucket to store the output

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))


# In[78]:


# This code is used to get the training container of sagemaker built-in algorithms
# all we have to do is to specify the name of the algorithm, that we want to use

# Let's obtain a reference to the XGBoost container image
# Note that all regression models are named estimators
# You don't have to specify (hardcode) the region, get_image_uri will get the current region name using boto3.Session

from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(boto3.Session().region_name, 'xgboost','0.90-2') # Latest version of XGboost


# In[79]:


# Specify the type of instance that we would like to use for training 
# output path and sagemaker session into the Estimator. 
# We can also specify how many instances we would like to use for training

# Recall that XGBoost works by combining an ensemble of weak models to generate accurate/robust results. 
# The weak models are randomized to avoid overfitting

# num_round: The number of rounds to run the training.


# Alpha: L1 regularization term on weights. Increasing this value makes models more conservative.

# colsample_by_tree: fraction of features that will be used to train each tree.

# eta: Step size shrinkage used in updates to prevent overfitting. 
# After each boosting step, eta parameter shrinks the feature weights to make the boosting process more conservative.


Xgboost_regressor1 = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.m5.2xlarge',
                                       output_path = output_location,
                                       sagemaker_session = sagemaker_session)

#We can tune the hyper-parameters to improve the performance of the model

Xgboost_regressor1.set_hyperparameters(max_depth = 10,
                           objective = 'reg:linear',
                           colsample_bytree = 0.3,
                           alpha = 10,
                           eta = 0.1,
                           num_round = 100
                           )


# In[80]:


# Creating "train", "validation" channels to feed in the model
# Source: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html

train_input = sagemaker.session.s3_input(s3_data = s3_train_data, content_type='csv',s3_data_type = 'S3Prefix')
valid_input = sagemaker.session.s3_input(s3_data = s3_validation_data, content_type='csv',s3_data_type = 'S3Prefix')


data_channels = {'train': train_input,'validation': valid_input}


Xgboost_regressor1.fit(data_channels)


# # TASK #11: DEPLOY THE MODEL TO MAKE PREDICTIONS

# In[107]:


# Deploy the model to perform inference 

Xgboost_regressor = Xgboost_regressor1.deploy(initial_instance_count = 1, instance_type = 'ml.m5.2xlarge')


# In[115]:


'''
Content type over-rides the data that will be passed to the deployed model, since the deployed model expects data
in text/csv format, we specify this as content -type.

Serializer accepts a single argument, the input data, and returns a sequence of bytes in the specified content
type

Reference: https://sagemaker.readthedocs.io/en/stable/predictors.html
'''
from sagemaker.predictor import csv_serializer, json_deserializer

Xgboost_regressor.content_type = 'text/csv'
Xgboost_regressor.serializer = csv_serializer
Xgboost_regressor.deserializer = None


# In[116]:


X_test.shape


# In[133]:


# making prediction

predictions1 = Xgboost_regressor.predict(X_test[0:10000])


# In[148]:


predictions2 = Xgboost_regressor.predict(X_test[10000:20000])


# In[149]:


predictions3 = Xgboost_regressor.predict(X_test[20000:30000])


# In[150]:


predictions4 = Xgboost_regressor.predict(X_test[30000:31618])


# In[168]:


predictions4


# In[151]:


# custom code to convert the values in bytes format to array

def bytes_2_array(x):
    
    # makes entire prediction as string and splits based on ','
    l = str(x).split(',')
    
    # Since the first element contains unwanted characters like (b,',') we remove them
    l[0] = l[0][2:]
    #same-thing as above remove the unwanted last character (')
    l[-1] = l[-1][:-1]
    
    # iterating through the list of strings and converting them into float type
    for i in range(len(l)):
        l[i] = float(l[i])
        
    # converting the list into array
    l = np.array(l).astype('float32')
    
    # reshape one-dimensional array to two-dimensional array
    return l.reshape(-1,1)
    


# In[152]:


predicted_values_1 = bytes_2_array(predictions1)


# In[153]:


predicted_values_1.shape


# In[154]:


predicted_values_2 = bytes_2_array(predictions2)
predicted_values_2.shape


# In[155]:


predicted_values_3 = bytes_2_array(predictions3)
predicted_values_3.shape


# In[156]:


predicted_values_4 = bytes_2_array(predictions4)
predicted_values_4.shape


# In[163]:


predicted_values = np.concatenate((predicted_values_1, predicted_values_2, predicted_values_3, predicted_values_4))


# In[165]:


predicted_values.shape


# In[166]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test, predicted_values)),'.3f'))
MSE = mean_squared_error(y_test, predicted_values)
MAE = mean_absolute_error(y_test, predicted_values)
r2 = r2_score(y_test, predicted_values)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 


# In[167]:


# Delete the end-point

Xgboost_regressor.delete_endpoint()


# # TASK #12: PERFORM HYPERPARAMETERS OPTIMIZATION

# See Slides for detailed steps

# # TASK #13: TRAIN THE MODEL WITH BEST PARAMETERS

# In[190]:


# We have pass in the container, the type of instance that we would like to use for training 
# output path and sagemaker session into the Estimator. 
# We can also specify how many instances we would like to use for training

Xgboost_regressor = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.m4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sagemaker_session)

# We can tune the hyper-parameters to improve the performance of the model
Xgboost_regressor.set_hyperparameters(max_depth=25,
                           objective='reg:linear',
                           colsample_bytree = 0.3913546819101119,
                           alpha = 1.0994354985124635,
                           eta = 0.23848185159806115,
                           num_round = 237
                           )


# In[191]:


train_input = sagemaker.session.s3_input(s3_data = s3_train_data, content_type='csv',s3_data_type = 'S3Prefix')
valid_input = sagemaker.session.s3_input(s3_data = s3_validation_data, content_type='csv',s3_data_type = 'S3Prefix')
data_channels = {'train': train_input,'validation': valid_input}
Xgboost_regressor.fit(data_channels)


# In[192]:


# Deploying the model to perform inference

Xgboost_regressor = Xgboost_regressor.deploy(initial_instance_count = 1,
                                          instance_type = 'ml.m4.xlarge')


# In[194]:


from sagemaker.predictor import csv_serializer, json_deserializer

Xgboost_regressor.content_type = 'text/csv'
Xgboost_regressor.serializer = csv_serializer
Xgboost_regressor.deserializer = None


# In[ ]:


# Try to make inference with the entire testing dataset (Crashes!)
predictions = Xgboost_regressor.predict(X_test)
predicted_values = bytes_2_array(predictions)


# In[196]:


predictions1 = Xgboost_regressor.predict(X_test[0:10000])


# In[197]:


predicted_values_1 = bytes_2_array(predictions1)
predicted_values_1.shape


# In[198]:


predictions2 = Xgboost_regressor.predict(X_test[10000:20000])
predicted_values_2 = bytes_2_array(predictions2)
predicted_values_2.shape


# In[199]:


predictions3 = Xgboost_regressor.predict(X_test[20000:30000])
predicted_values_3 = bytes_2_array(predictions3)
predicted_values_3.shape


# In[200]:


predictions4 = Xgboost_regressor.predict(X_test[30000:31618])
predicted_values_4 = bytes_2_array(predictions4)
predicted_values_4.shape


# In[201]:


predicted_values = np.concatenate((predicted_values_1, predicted_values_2, predicted_values_3, predicted_values_4))


# In[202]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test, predicted_values)),'.3f'))
MSE = mean_squared_error(y_test, predicted_values)
MAE = mean_absolute_error(y_test, predicted_values)
r2 = r2_score(y_test, predicted_values)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 


# In[203]:


# Delete the end-point

Xgboost_regressor.delete_endpoint()


# # EXCELLENT JOB! 

# # MINI CHALLENGE SOLUTIONS

# In[ ]:


feature.info()
feature.describe()
sales.info()
sales.describe()
stores.info()
stores.describe()


# In[ ]:


def get_month(x):
    return int(str(x).split('-')[1])

df['month'] = df['Date'].apply(get_month)


# In[ ]:


df['IsHoliday'] = df['IsHoliday'].replace({True : 1, False : 0})

