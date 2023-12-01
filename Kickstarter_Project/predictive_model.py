
# -*- coding: utf-8 -*-

# Load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

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
kickstarter_df = pd.read_excel('kickstarter.xlsx')

# Preprocessing --------------------------------------------------

# Copy the DataFrame for preprocessing
df1 = kickstarter_df.copy()

# Keep only successful/failed projects
df1 = df1[kickstarter_df['state'].str.contains('successful|failed', na=False)]

# Null Handling
df1.isnull().sum()
df1.drop(columns='launch_to_state_change_days', axis=1, inplace=True)

# Create region column based on currency
df1['region'] = df1.apply(currency_to_region, axis=1)

# Category Column Null Handling
temp = df1.isnull().any(axis=1)
df2 = df1[~temp]
df2.info()
df2.isnull().sum()

# Create a DataFrame with only null in category column
df3 = df1[temp]

# Fill NAs in category column with 'categoryless'
df4 = df1.copy()
df4.category.fillna('categoryless', inplace=True)
df4[temp].category

# Define variables
y = df2['state']
predictors = ['goal', 'category', 'region', 'name_len_clean', 'name_len', 'created_at_hr',
              'created_at_weekday']
X = df2[predictors]
X = pd.get_dummies(X, columns=['region', 'category', 'created_at_weekday'])

columns = X.columns

# Standardization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Training GBT
optimal_i, best_score = 0, 0
print("\nFinding optimal max_features parameter:\n")
for i in range(2, 10):
    model3 = GradientBoostingClassifier(random_state=0, max_features=i, n_estimators=100)

    # K-fold cross-validation
    scores = cross_val_score(model3, X=X, y=y, cv=5)
    x = np.average(scores)
    print(i, x)
    if x > best_score:
        best_score, optimal_i = x, i

# Find optimal n_estimators
optimal_j, best_score = 0, 0
print("\nFinding optimal n_estimators parameter:\n")
for j in range(90, 110):
    model3 = GradientBoostingClassifier(random_state=0, max_features=optimal_i, n_estimators=j)

    # K-fold cross-validation
    scores = cross_val_score(model3, X=X, y=y, cv=5)
    x = np.average(scores)
    print(j, x)
    if x > best_score:
        best_score, optimal_j = x, j

# Find optimal min_samples_split
optimal_k, best_score = 0, 0
print("\nFinding optimal min_samples_split parameter:\n")
for k in range(2, 10):
    model3 = GradientBoostingClassifier(random_state=0, max_features=optimal_i, n_estimators=optimal_j,
                                        min_samples_split=k)

    # K-fold cross-validation
    scores = cross_val_score(model3, X=X, y=y, cv=5)
    x = np.average(scores)
    print(k, x)
    if x > best_score:
        best_score, optimal_k = x, k

# Find Accuracy of optimized model.
opt_model = GradientBoostingClassifier(random_state=0, max_features=optimal_i, n_estimators=optimal_j,
                                       min_samples_split=optimal_k)
train_scores = cross_val_score(opt_model, X=X, y=y, cv=5)
print(f'Accuracy is {np.average(train_scores)} with max_features={optimal_i}, n_estimators={optimal_j} and min_samples_split = {optimal_k}')

# Fit the model
opt_model.fit(X, y)

# Feature Importance
opt_model.feature_importances_
print(pd.DataFrame(list(zip(columns, opt_model.feature_importances_)), columns=['predictor', 'coefficient']))

#%% Grading set
grade_df = pd.read_excel("Kickstarter-Grading-Sample.xlsx")

# Preprocessing
grade_df1 = grade_df.copy()

# Keep only successful/failed projects
grade_df1 = grade_df1[grade_df1['state'].str.contains('successful|failed', na=False)]

# Null Handling
grade_df1.isnull().sum()
grade_df1.drop(columns='launch_to_state_change_days', axis=1, inplace=True)

# Create region column from currency
grade_df1['region'] = df1.apply(currency_to_region, axis=1)

# Category Column Null Handling
grade_temp = df1.isnull().any(axis=1)
grade_df2 = df1[~grade_temp]

# Define variables
y_grade = grade_df2['state']
predictors = ['goal', 'category', 'region', 'name_len_clean', 'name_len', 'created_at_hr',
              'created_at_weekday']
X_grade = grade_df2[predictors]
X_grade = pd.get_dummies(X_grade, columns=['region', 'category', 'created_at_weekday'])

# Run model on Grading data
y_grade_pred = opt_model.predict(X_grade.values)

# Print Accuracy
print(accuracy_score(y_grade, y_grade_pred))
