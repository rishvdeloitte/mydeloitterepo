#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\mlp\Big Mart Sale Prediction\Train.csv')
train.head()
train.shape

test = pd.read_csv(r'C:\Users\naveen chauhan\Desktop\mldata\mlp\Big Mart Sale Prediction\Test.csv')
test.head()
test.shape

train.isnull().sum()
test.isnull().sum()

train.describe()

correl = train.corr()
ax = plt.subplots(figsize=(15, 9))
sns.heatmap(correl, vmax=0.8, square=True)

train.head()
train.Item_Fat_Content.value_counts()
train.Item_Type.value_counts()
train.Outlet_Identifier.value_counts()
train.Outlet_Size.value_counts()
train.Outlet_Location_Type.value_counts()
train.Outlet_Type.value_counts()
train.head()
train.isnull().sum()

train.Item_Weight.hist(bins=50)
train.Outlet_Size.hist(bins=50)
train.Outlet_Size.value_counts()
Item_Sales = train.Item_Outlet_Sales
data = train.append(test)
data.shape
data.isnull().sum()
data.isnull().sum()
correlation = data.corr()
sns.heatmap(correlation, vmax=0.8, square=True)
data.apply(lambda x: len(x.unique()))
data.dtypes
data.dtypes.index
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x] == 'object']
categorical_columns
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier', 'Outlet_Identifier']]
categorical_columns
for col in categorical_columns:
    print('frequency of categories for variable')
    print(data[col].value_counts())

data.Item_Weight.fillna(data.Item_Weight.mean(), inplace=True)
from scipy.stats import mode
data.Outlet_Size = data.Outlet_Size.map({'Small': 0, 'Medium': 1, 'High': 2})
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: mode(x).mode[0]))
miss_bool = data['Outlet_Size'].isnull()
data.loc[miss_bool, 'Outlet_Size'] = data.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

for i in data.dtypes.index:
    if len(data[i].value_counts()) < 30:
        print(i, "\n", data[i].value_counts())

data.pivot_table(index='Outlet_Type', values='Item_Outlet_Sales')
data.Item_Visibility.hist(bins=50)
data.Item_Visibility.mean()
data.loc[data['Item_Visibility'] == 0, 'Item_Visibility'] = data.Item_Visibility.mean()
data.Item_Type.value_counts()
data['Item_Type_Combined'] = data.Item_Identifier.apply(lambda x: x[0:2])
data['Item_Type_Combined'].value_counts()
data['Item_Type_Combined'] = data.Item_Type_Combined.map({'FD': 'Food and Drinks', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
data['Item_Type_Combined'].value_counts()
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()
data.Item_Fat_Content.value_counts()
data.Item_Fat_Content = data.Item_Fat_Content.replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
data.Item_Fat_Content.value_counts()
data.loc[data['Item_Type_Combined'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
data.Item_Fat_Content.value_counts()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data['Outlet'] = lb.fit_transform(data['Outlet_Identifier'])
var = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size', 'Item_Type_Combined']
lb = LabelEncoder()
for item in var:
    data[item] = lb.fit_transform(data[item])
data.drop(['Outlet_Establishment_Year', 'Item_Type'], inplace=True, axis=1)

Item_Sales = data.Item_Outlet_Sales
train = data.iloc[:8523, :]
test = data.iloc[8523:, :]
test.drop('Item_Outlet_Sales', inplace=True, axis=1)

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier', 'Outlet_Identifier']

from sklearn import model_selection, metrics

def modelfit(alg, dtrain, dtest, predictor, target, IDcol, filename):
    alg.fit(dtrain[predictor], dtrain[target])
    prediction = alg.predict(dtrain[predictor])
    cv_score = model_selection.cross_val_score(alg, dtrain[predictor], dtrain[target], cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    print(np.sqrt(metrics.mean_squared_error(dtrain[target].values, prediction))
    print("CV_SCORE : mean - %.4g | std - %.4g | max - %.4g | min - %.4g" % (np.mean(cv_score), np.std(cv_score), np.max(cv_score), np.min(cv_score)))
    dtest[target] = alg.predict(dtest[predictor])
    IDcol.append(target)
    submission = pd.DataFrame({x: dtest[x] for x in IDcol})
    submission.to_csv("C:\\Users\\naveen chauhan\\Desktop\\mldata\\mlp\\Big Mart Sale Prediction\\" + filename, index=False)

from sklearn.linear_model import LinearRegression, Ridge, Lasso

predictor = [x for x in train.columns if x not in [target] + IDcol]
alg1 = LinearRegression()
modelfit(alg1, train, test, predictor, target, IDcol, 'alg1.csv')

predictors = [x for x in train.columns if x not in [target] + IDcol]
alg2 = Ridge(alpha=0.05, normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target] + IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

predictors = ['Item_MRP', 'Outlet_Type', 'Outlet', 'Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')

predictors = ['Item_MRP', 'Outlet_Type', 'Outlet', 'Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')



# In[ ]:





# In[ ]:




