#!/usr/bin/env python
# coding: utf-8

# Prediction and Clustering Using Unsupervised ML Algorithm (K- Means)

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[43]:


# Read the CSV file
iris_df = pd.read_csv("C:\\Users\\JOY\\Downloads\\Iris (1).csv")


# In[44]:


iris_df    #showing the whole dataset 


# In[45]:


iris_df['Species'].value_counts()      #cheking the number of samples of each 


# In[46]:


iris_df.info ()
                  #getting the info of the dataset 


# In[47]:


iris_df.describe() # getting statistical measures of the dataset 


# In[48]:


iris_df.isnull().sum()


# In[49]:


# Drop the "Id" column as it is not relevant for clustering
iris_df = iris_df.drop("Id", axis=1)

# Separate the features (X) and the target variable (y)
X = iris_df.iloc[:, :-1].values

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[50]:


# Create an empty list to store the sum of squared distances (inertia) for each K value
inertia = []

# Iterate over a range of K values (from 1 to 10)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Sum of Squared Distances")
plt.title("Elbow Curve")
plt.show()


# In[51]:


# Set the chosen number of clusters
k = 3

# Instantiate and fit the K-means model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Get the cluster labels
labels = kmeans.labels_


# In[52]:


# Create a scatter plot for the first two features (SepalLengthCm and SepalWidthCm)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker="x", c="red")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("K-means Clustering (K=3)")
plt.show()


# In[ ]:





# In[53]:


from mpl_toolkits.mplot3d import Axes3D

# Create a 3D scatter plot using the first three features
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels, cmap="viridis")
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.set_zlabel("Petal Length (cm)")
ax.set_title("K-means Clustering (K=3) - 3D Scatter Plot")
plt.show()


# In[54]:


new_input = [[5.1, 3.5, 1.4, 0.2]]  # Example input, replace with your own data
new_input_scaled = scaler.transform(new_input)


# In[55]:


predicted_cluster = kmeans.predict(new_input_scaled)
print("Predicted Cluster:", predicted_cluster)


# In[60]:


import scipy.stats as stats

# Perform ANOVA test
fvalue, pvalue = stats.f_oneway(iris_df[iris_df['Species'] == 'Iris-setosa']['PetalLengthCm'],
                                iris_df[iris_df['Species'] == 'Iris-versicolor']['PetalLengthCm'],
                                iris_df[iris_df['Species'] == 'Iris-virginica']['PetalLengthCm'])

if pvalue < 0.05:
    print("There are significant differences in petal lengths between different species.")
else:
    print("There are no significant differences in petal lengths between different species.")


# In[61]:


# Calculate correlation coefficient
correlation = iris_df['SepalWidthCm'].corr(iris_df['PetalWidthCm'])

if correlation > 0:
    print("There is a positive correlation between sepal width and petal width.")
elif correlation < 0:
    print("There is a negative correlation between sepal width and petal width.")
else:
    print("There is no significant correlation between sepal width and petal width.")


# In[62]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data into features and target variable
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)


# In[67]:


import seaborn as sns

# Create a box plot to visualize the distribution of sepal length
sns.boxplot(x=iris_df['SepalLengthCm'])
plt.show()

# Calculate the upper and lower bounds for outliers using the Tukey method
Q1 = iris_df['SepalLengthCm'].quantile(0.25)
Q3 = iris_df['SepalLengthCm'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify the outliers
outliers = iris_df[(iris_df['SepalLengthCm'] < lower_bound) | (iris_df['SepalLengthCm'] > upper_bound)]
print("Outliers in sepal length measurements:\n", outliers)

The 5 infernces which one may wish to draw is stated below.

1. Are there significant differences in petal lengths between different species?
   Insight: By performing analysis of variance (ANOVA) or t-tests, we can determine if there are statistically significant     differences in petal lengths among different species. This can help us understand the distinctive characteristics of each species based on their petal lengths.

2. Is there a relationship between sepal width and petal width?
Insight: By calculating the correlation coefficient between sepal width and petal width, we can determine if there is a linear relationship between these two variables. This can provide insights into the co-variation of these attributes in the Iris dataset.

3. Can we accurately predict the species of an iris based on its sepal and petal dimensions?
Insight: By training a supervised machine learning model (e.g., logistic regression, decision tree, or support vector machine) using the sepal and petal dimensions as input features and the species as the target variable, we can evaluate the model's accuracy in predicting the species of an iris based on its dimensions. This can help us understand the predictive power of these attributes.

4. Are there any outliers in the sepal length measurements?
Insight: By visually inspecting box plots or using statistical methods such as the Tukey method or z-scores, we can identify any outliers in the sepal length measurements. Outliers may indicate measurement errors or unique instances in the dataset, and understanding them can help ensure data quality.

5. How well does the K-means clustering algorithm separate the different species?
Insight: By visually comparing the K-means clustering results with the actual species labels, we can assess the effectiveness of the algorithm in separating the different species. This evaluation can help us understand the clustering performance and validate the algorithm's ability to group similar instances together.
# # Run and Executed by Joy Chatterjee
