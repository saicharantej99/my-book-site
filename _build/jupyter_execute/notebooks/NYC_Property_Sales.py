#!/usr/bin/env python
# coding: utf-8

# <img src="https://shwetkm.github.io/upxlogo.png"></img>
# 
# 
# # Case study 2: Linear regression model to predict sales price of properties

# ## Business Objective 
# 
# This dataset contains properties sold in New York City over a 12-month period from September 2016 to September 2017. The objective is to build a model to predict sale value in the future.

# ## Load Dataset

# In[1]:


#Download dataset
#!wget -q https://www.dropbox.com/s/6tc7e6rc395c7jz/nyc-property-sales.zip


# In[2]:


#Unzip the data
#!unzip nyc-property-sales.zip > /dev/null; echo " done."


# In[3]:


#!ls


# In[4]:


#Install Packages

#!pip -q install plotly-express
#!pip -q install shap
#!pip -q install eli5
#!pip -q install lime


# ## Import Packages

# In[5]:


#Import basic packages

import warnings
warnings.filterwarnings("ignore")
import time
import pandas as pd               
import numpy as np
import pickle

from sklearn.model_selection import train_test_split   #splitting data
from pylab import rcParams
from sklearn.linear_model import LinearRegression         #linear regression
from sklearn.metrics.regression import mean_squared_error #error metrics
from sklearn.metrics import mean_absolute_error

import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)



# In[6]:


# Code for displaying plotly express plots inline in colab
def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))
  
import plotly_express as px


# ## Read Data
# 
# 

# ### 1. Through Pandas

# In[7]:


# Read data through Pandas and compute time taken to read

t_start = time.time()
df_prop = pd.read_csv('nyc-rolling-sales.csv')
t_end = time.time()
print('pd.read_csv(): {} s'.format(t_end-t_start)) # time [s]
df_prop.head()


# ### 2. Through Dask

# In[8]:


# Read data through Dask and compute time taken to read

import dask.dataframe as dd

t_start = time.time()
df_dask = dd.read_csv('nyc-rolling-sales.csv')
t_end = time.time()
print('dd.read_csv(): {} s'.format(t_end-t_start)) # time [s]
df_dask.tail()


# **Observation**: Dask is almost 10 times faster than Pandas when reading files.

# ### 3. Subsample into n rows

# In[9]:


df_prop.shape


# In[10]:


# Sample n rows
n = df_prop.shape[0]
df_prop = df_prop.sample(n)
df_prop.shape


# ## Exploratory Data Analysis

# ### Let's look into summary statistics

# In[11]:


#Let's look into the total number of columns and observations in the dataset
df_prop.info()


# In[12]:


df_prop.isnull().sum()


# In[13]:


#Let's look into summary statistics of data
df_prop.describe()


# **Observation:** 
# - There is a column called Unnamed: 0 which is not required as it contains only continuous index numbers
# - The datatypes of saleprice is not correct because the summary statistics of sale price is not displayed
# 
# Hence there is a lot of data cleaning to perform.

# ### Data Cleaning

# #### Pandas profiling

# In[14]:


'''#Perform Pandas profiling to understand quick overview of columns

import pandas_profiling
report = pandas_profiling.ProfileReport(df_prop)
#covert profile report as html file
report.to_file("property_data.html")'''


# #### Removal of unnecessary columns

# **Observation:**
# - From Pandas profiling we understand EASEMENT column has no significant value and thus has to be removed.

# In[15]:


# Let's explore why EASE-MENT has to be rejected
df_prop['EASE-MENT'].unique()


# In[16]:


#This column has no significance other than being an iterator
del df_prop['Unnamed: 0']
#This column has no significant value
del df_prop['EASE-MENT']


# **Observation:**
# - From Pandas profiling we understand SALE PRICEcolumns have string value in some rows and thus has to be removed.
# - From Pandas Profiling we understand LAND SQUARE FEET and GROSS SQUARE FEET columns have string values which have to replaced by appropriate values 

# In[17]:


df_prop['SALE PRICE'] = df_prop['SALE PRICE'].replace(' -  ',np.nan)
df_prop.dropna(inplace=True)


# In[18]:


df_prop.shape


# In[19]:


df_prop['LAND SQUARE FEET'] = df_prop['LAND SQUARE FEET'].replace(' -  ',np.nan)
df_prop['GROSS SQUARE FEET'] = df_prop['GROSS SQUARE FEET'].replace(' -  ',np.nan)


# In[20]:


# count the number of NaN values in each column
print(df_prop.isnull().sum())


# In[21]:


df_prop['LAND SQUARE FEET'] = df_prop['LAND SQUARE FEET'].replace('0',np.nan)
df_prop['GROSS SQUARE FEET'] = df_prop['GROSS SQUARE FEET'].replace('0',np.nan)


# In[22]:


print(df_prop.isnull().sum())


# In[23]:


df_prop.describe()


# In[24]:


## Define a function impute_median and fill land square feet and gross square feet with median values
def impute_median(series):
    return series.fillna(series.median())

df_prop['LAND SQUARE FEET'] = df_prop['LAND SQUARE FEET'].transform(impute_median)
df_prop['GROSS SQUARE FEET'] = df_prop['GROSS SQUARE FEET'].transform(impute_median)


# In[25]:


df_prop.info()


# In[26]:


df_prop.describe()


# In[27]:


#Convert few column datatypes into appropriate ones for conserving memory

df_prop['TAX CLASS AT TIME OF SALE'] = df_prop['TAX CLASS AT TIME OF SALE'].astype('category')
df_prop['TAX CLASS AT PRESENT'] = df_prop['TAX CLASS AT PRESENT'].astype('category')
df_prop['LAND SQUARE FEET'] = pd.to_numeric(df_prop['LAND SQUARE FEET'], errors='coerce')
df_prop['GROSS SQUARE FEET']= pd.to_numeric(df_prop['GROSS SQUARE FEET'], errors='coerce')
df_prop['SALE PRICE'] = pd.to_numeric(df_prop['SALE PRICE'], errors='coerce')
df_prop['BOROUGH'] = df_prop['BOROUGH'].astype('category')


# In[28]:


#The datatypes have now been changed
df_prop.info()


# In[29]:


df_prop.info()


# In[30]:


# Let's remove sale price with a nonsensically small dollar amount: $0 most commonly. 
# Since these sales are actually transfers of deeds between parties: for example, parents transferring ownership to their home to a child after moving out for retirement.

df_prop = df_prop[df_prop['SALE PRICE']!=0]


# In[31]:


#Let's also remove observations that have gross square feet less than 400 sq. ft
#Let's also remove observations that have gross square feet less than 400 sq. ft
#Let's also remove observations that have sale price than 1000 dollars

df_prop = df_prop[df_prop['GROSS SQUARE FEET']>400]
df_prop = df_prop[df_prop['LAND SQUARE FEET']>400]
df_prop = df_prop[df_prop['SALE PRICE']>1000]


# In[32]:


df_prop.describe()


# In[33]:


df_prop[df_prop['SALE PRICE']==2210000000]


# **Observation:** The most expensive property in NYC is a whopping 2 billion dollars which can be considered as an outlier.

# #### Let's remove outiers!

# In[34]:


q = df_prop["SALE PRICE"].quantile(0.99)
q


# In[35]:


df_prop = df_prop[df_prop["SALE PRICE"] < q]
df_prop_lin = df_prop.copy()


# In[36]:


# Convert sale date into time,month,year and day
df_prop['SALE DATE']=pd.to_datetime(df_prop['SALE DATE'])
df_prop['year']=df_prop['SALE DATE'].dt.year
df_prop['month']=df_prop['SALE DATE'].dt.month
df_prop['day']=df_prop['SALE DATE'].dt.day
df_prop['time']=df_prop['SALE DATE'].dt.hour
df_prop['day_week']=df_prop['SALE DATE'].dt.weekday_name


# In[37]:


df_prop.head()


# In[38]:


df_prop.info()


# ### Data Visualization

# In[39]:


#Assign numbered bouroughs to bourough names
dic = {1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5:'Staten Island'}
df_prop["borough_name"] = df_prop["BOROUGH"].apply(lambda x: dic[x])


# #### Count of properties in NYC in each bororugh

# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
df_prop.borough_name.value_counts().nlargest().plot(kind='bar', figsize=(10,5))
plt.title("Number of properties by city")
plt.ylabel('Number of properties')
plt.xlabel('City');
plt.show()


# #### Distribution of Sale Price

# In[41]:


df_prop['SALE PRICE'].describe()


# **Observation**: The maximum sale price is 14 million

# In[42]:


df_prop['SALE PRICE'].plot.hist(bins=20, figsize=(12, 6), edgecolor = 'white')
plt.xlabel('price', fontsize=12)
plt.title('Price Distribution', fontsize=12)
plt.show()


# **Observation**: The distribution is highly skewed towards the right which implies there are **lesser properties that have a very high prices**.  
# 

# In[43]:


sns.boxplot(df_prop["SALE PRICE"])


# In[44]:


df_prop["log_price"] = np.log(df_prop["SALE PRICE"] + 1)
sns.boxplot(df_prop.log_price)


# #### Correlation between selected variables
# 
# The heat map produces a correlation plot between variables of the dataframe.

# In[45]:


plt.figure(figsize=(15,10))
c = df_prop[df_prop.columns.values[0:19]].corr()
sns.heatmap(c,cmap="BrBG",annot=True)


# **Observation**: The heat map illustrates that sale price is independent of all column values that could be considered for linear regression.

# #### Explore how gross square feet affects sale price 

# In[46]:


configure_plotly_browser_state()
px.scatter(df_prop, x="GROSS SQUARE FEET", y="SALE PRICE", size ="TOTAL UNITS" ,color="borough_name",
           hover_data=["BUILDING CLASS CATEGORY","LOT"], log_x=True, size_max=60)


# **Observation:** 
# 
# - Properties with more total units do not fetch larger sales price
# - Properties in Staten Island have comparitively lesser sales price in comparison with other boroughs in New york city

# #### Explore how tax class at the time of sale affect sales price

# In[47]:


configure_plotly_browser_state()
px.box(df_prop, x="borough_name", y="SALE PRICE", color="TAX CLASS AT TIME OF SALE",hover_data=['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY'],notched=True)


# **Observation:** 
# 
# - Manhatten has the highest priced properties that have a tax class 1 representing  residential property of up to three units (such as one-,two-, and three-family homes and small stores or offices with one or two attached apartments) as compared to other boroughs. 
# - Properties in Staten Island have comparitively lesser sales price in comparison with other boroughs in New york city

# In[48]:


configure_plotly_browser_state()
px.box(df_prop, x="day_week", y="SALE PRICE", color="TAX CLASS AT TIME OF SALE", notched=True)


# **Observation:** 
# 
# - On Saturdays there are no sales for the tax class 4 which represents properties such as  such as offices, factories, warehouses, garage buildings, etc. 

# ## Model Building

# ###  Prepare the Data for model building

# #### Delete columns not necessary for prediction

# In[49]:


df_prop_lin.columns


# In[50]:


#Dropping few columns
del df_prop_lin['BUILDING CLASS AT PRESENT']
del df_prop_lin['BUILDING CLASS AT TIME OF SALE']
del df_prop_lin['NEIGHBORHOOD']
del df_prop_lin['ADDRESS']
del df_prop_lin['SALE DATE']
del df_prop_lin['APARTMENT NUMBER']
del df_prop_lin['RESIDENTIAL UNITS']
del df_prop_lin['COMMERCIAL UNITS']


# In[51]:


df_prop_lin.info()


# In[52]:


df_prop_lin.head()


# In[53]:


season = [winter, rainy, summer]
 season_rainy, season_summer


# #### Perform one-hot encoding for categorical variables

# In[54]:


#Select the variables to be one-hot encoded
one_hot_features = ['BOROUGH', 'BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE']
# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(df_prop_lin[one_hot_features],drop_first=True)
one_hot_encoded
#one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)


# In[55]:


df_prop_lin.info()


# In[56]:


one_hot_encoded.info()


# In[57]:


# Replacing categorical columns with dummies
fdf = df_prop_lin.drop(one_hot_features,axis=1)
fdf = pd.concat([fdf, one_hot_encoded] ,axis=1)


# In[69]:


fdf.info()
#print (fdf.shape)


# In[70]:


del fdf['LOT']
del fdf['BLOCK']
del fdf['ZIP CODE']
del fdf['YEAR BUILT']


# In[87]:


fdf.info()


# #### Training and Test data

# In[122]:


X = fdf.drop(['SALE PRICE'],axis=1)
print (X)
y = fdf['SALE PRICE']

fdf_normalized = pd.DataFrame(data=X, index=X.index, columns=X.columns)

print (y)


# In[123]:


fdf_normalized.head(6)


# In[ ]:





# #### Split the data into train and test

# In[124]:


X_train, X_test, y_train, y_test = train_test_split(fdf_normalized,y)


# In[125]:


X_test.shape


# In[126]:


X_train.shape


# ### Train the model

# In[127]:


# initialize the model
lr= LinearRegression()

# fit the model
model_fit=lr.fit(X_train,y_train)
print (model_fit.coef_)
print (model_fit.intercept_)
y= 3*Area + 7*LandSqfeet - 9xTax_Class+16


# ### Test the model

# In[208]:


#predict on test data
test_pred = model_fit.predict(X_test)
#Answers provided by us in the exam
#y_test => Answer Key
test_pred


# In[129]:


test_null = y_test.mean()
#mean squared error
mse=mean_squared_error(y_test,test_null)

#root mean squared error
print('test rmse: {}'.format(np.sqrt(mse)))

#mean absolute error
#mae=mean_absolute_error(y_train,train_pred)
#print('train mae: {}'.format(mae))


# In[212]:


test_null = np.zeros_like(y_test, dtype=float)
test_null.fill(y_test.mean())
mse=mean_squared_error(y_test,test_null)
print('Null rmse: {}'.format(np.sqrt(mse)))


# In[130]:


len(test_pred)


# ## Model Explainability

# ### LIME

# In[131]:


X_test.values[4].shape


# In[132]:


# Import lime package
import lime
import lime.lime_tabular

#Find caegorical features
categorical_features = np.argwhere(np.array([len(set(X_test.values[:,x])) for x in range(X_test.values.shape[1])]) <= 10).flatten()
#Lime explainer for regression
explainer = lime.lime_tabular.LimeTabularExplainer(X_test.values,
feature_names=X_test.columns.values.tolist(),
class_names=['PriceCrore'],
categorical_features=categorical_features,
verbose=True, mode='regression')
ind = 4
#Fit on test data
exp = explainer.explain_instance(X_test.values[ind], model_fit.predict, num_features=6)
#Show in notebook features influencing predictions
exp.show_in_notebook(show_table=True)


# ### ELI5

# In[133]:


# Import Eli5 package
import eli5
from eli5.sklearn import PermutationImportance

# Find the importance of columns for prediction
perm = PermutationImportance(model_fit, random_state=1).fit(X_test,test_pred)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[134]:


#Understanding how each feature influences the prediction
eli5.show_prediction(model_fit, doc=X_test.iloc[[ind]], feature_names=list(X_test.columns))
Wage = 1*Education + 300000

Wage = 300002
Wage = 300003
Wage = 300040


# ### SHAP

# In[135]:


#Import SHAP package
import shap

#Create explainer for linear model
explainer = shap.LinearExplainer(model_fit,data=X_test.values)
shap_values = explainer.shap_values(X_test)


# In[136]:


#Understanding how each feature influences the prediction

shap.initjs()
ind = 11


shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test.iloc[ind,:],
    feature_names=X_test.columns.tolist()
)


# In[137]:



shap.summary_plot(shap_values,X_test)


# In[ ]:




