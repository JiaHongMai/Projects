# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:33:37 2023

@author: Reydarz
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:00:24 2022

@author: mjiah
"""

# Load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def currency_to_region(df):
    """
    Map currency to a geographical region.

    Parameters:
    - df (DataFrame): DataFrame containing a 'currency' column.

    Returns:
    - str: Geographical region corresponding to the currency.
    """
    if df.currency in ['EUR', 'CHF']:
        return 'Europe'
    elif df.currency in ['USD', 'CAD', 'MXN']:
        return 'North America'
    elif df.currency == 'GBP':
        return 'Great Britain'
    elif df.currency in ['AUD', 'NZD']:
        return 'Oceania'
    elif df.currency in ['SEK', 'DKK', 'NOK']:
        return 'Scandinavia'
    elif df.currency in ['SGD', 'HKD']:
        return 'Asia'

# Load dataset
kickstarter_df = pd.read_excel(r"Kickstarter.xlsx")

#%% Preprocessing 
df1 = kickstarter_df.copy()

# Keep only successful/failed
df1 = df1[kickstarter_df['state'].str.contains('successful|failed', na=False)]

# Null Handling
df1.isnull().sum()
df1.drop(columns='launch_to_state_change_days', axis=1, inplace=True)

# Create region column based on currency
df1['region'] = df1.apply(currency_to_region, axis=1)

# Category Column Null Handling
temp = df1.isnull().any(axis=1)

# Contains no Null in category column
df2 = df1[~temp] 
df2.info()
df2.isnull().sum()
df2['created_at_yr']

# Contains only null in category column
df3 = df1[temp] 

# Fill NAs with value 'categoryless'
df4 = df1.copy()
df4.category.fillna('categoryless', inplace=True)
df4[temp].category

#%% Define variables
y = df2['state']
predictors = ['goal', 'category', 'region']
X = df2[predictors]
X = pd.get_dummies(X, columns=['region', 'category'])

# Standardization
standardizer = MinMaxScaler()
X_std = standardizer.fit_transform(X)

#%% Run Kmeans 
kmeans = KMeans(n_clusters=3)
model11 = kmeans.fit(X_std)
labels = model11.predict(X_std)

# Cluster Membership 
X['ClusterMembership'] = labels 
cluster_0 = round(X[X.ClusterMembership == 0].mean(), 2)
cluster_1 = round(X[X.ClusterMembership == 1].mean(), 2)
cluster_2 = round(X[X.ClusterMembership == 2].mean(), 2)

# Save clusters to Excel
clusters = pd.concat([cluster_0, cluster_1, cluster_2], axis=1)
clusters.to_excel("Clusters_3.xlsx")

# Silhouette score
print(silhouette_score(X_std, labels))
