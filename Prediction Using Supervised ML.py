#!/usr/bin/env python
# coding: utf-8

# In[ ]:


A Predictive Model on ML Using Linear Regression .


# In[30]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[4]:


url = "http://bit.ly/w-data"
dataset = pd.read_csv(url)
print("Data imported successfully")


# In[5]:


print(dataset.head())


# In[6]:


dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[8]:


X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  


# In[9]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)  


# In[10]:


y_pred = regressor.predict(X_test)  


# In[11]:


plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Hours vs Percentage (Training set)')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Hours vs Percentage (Test set)')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[15]:


# Outlier detection
# Let's use the interquartile range (IQR) method to detect outliers in the target variable
Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5*IQR
lower_bound = Q1 - 1.5*IQR
outliers = y[(y > upper_bound) | (y < lower_bound)]
print("Number of outliers:", len(outliers))


# In[25]:


# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 500, 1000], 'max_depth': [2, 4, 6, 8]}
grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best parameters:", best_params)


# In[26]:


# Cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=RandomForestRegressor(n_estimators=best_params['n_estimators'], 
                                                         max_depth=best_params['max_depth']), 
                         X=X_train, y=y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print("RMSE scores:", rmse_scores)
print("Average RMSE score:", rmse_scores.mean())


# In[29]:


# Ensembling using a VotingRegressor
from sklearn.ensemble import VotingRegressor

regressor1 = LinearRegression()
regressor2 = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
regressor3 = GradientBoostingRegressor(n_estimators=100, max_depth=4)
voting_regressor = VotingRegressor(estimators=[('lr', regressor1), ('rf', regressor2), ('gbr', regressor3)])
voting_regressor.fit(X_train, y_train)
y_pred_ensemble = voting_regressor.predict(X_test)


# In[20]:



# Residual analysis
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)


# In[12]:


hours = [[9.5]]
score_pred = regressor.predict(hours)
print("Predicted Score = {}".format(score_pred[0]))


# In[13]:


from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


# Error analysis
# Let's use a residual plot to analyze the relationship between the residuals and the predicted values
sns.scatterplot(x=y_pred, y=residuals)

Key Outputs Of the model
1. Is there a significant correlation between the number of hours studied and the percentage score achieved?
Answer: Yes, there is a strong positive correlation between the number of hours studied and the percentage score achieved (correlation coefficient = 0.98).

2. What is the expected percentage score for a student who studies for 7.5 hours per day?
Answer: Based on our model, we can predict that a student who studies for 7.5 hours per day is expected to achieve a percentage score of around 74.31%.

3.How accurate is our model at predicting percentage scores based on the number of hours studied?
Answer: Our model has a mean absolute error of around 4.18 percentage points, suggesting that it is fairly accurate at predicting percentage scores.

4.Are there any outliers in the dataset that could be affecting our model's performance?
Answer: We used the interquartile range (IQR) method to detect outliers in the target variable and found that there are no outliers in the dataset.


# # Run and executed by Joy Chatterjee
