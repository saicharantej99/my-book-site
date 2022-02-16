#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

# load dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# normalize data
df_norm = (df - df.mean()) / df.std()

# PCA
pca = PCA(n_components=4)
pca.fit_transform(df_norm.values)
print (pca.explained_variance_ratio_)
print (iris.feature_names)
print (pca.explained_variance_)
variance_ratio_cum_sum=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(variance_ratio_cum_sum)
print (pca.components_)


# In[ ]:


PC1 = 0.5 * Sepal_length - 0.2 * Sepal_width + 0.58* Petal_length + 0.56 * Petal_width


# In[6]:


# PCA with 2 components
pca = PCA(n_components=2)
pca.fit_transform(df_norm)

# Dump components relations with features:
print (pd.DataFrame(pca.components_,columns=df_norm.columns,index = ['PC-1','PC-2']))


# In[7]:


import numpy as np
variance_ratio_cum_sum=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(variance_ratio_cum_sum)


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#Visualize coefficients using heat map
features=iris.feature_names
plt.figure(figsize=[25,5])
sns.heatmap(pca.components_[0:2,:],annot=True,cmap='viridis')
plt.yticks([0,1],["First component","Second component"],rotation=360,ha="right")
plt.xticks(range(len(features)),features,rotation=60,ha="left")
plt.xlabel("Feature")
plt.ylabel("Principal components")


# In[ ]:




