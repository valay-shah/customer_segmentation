# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing dataset
dataset = pd.read_csv(os.getcwd()+"/customer-segmentation-dataset/Mall_Customers.csv")
dataset_1 = dataset[['Gender','Age','Annual Income (k$)','Spending Score (1-100)']]
#print(dataset_1)
X = dataset.iloc[:, [3,4]].values

corr_matrix = dataset_1.corr()
print(corr_matrix['Annual Income (k$)'].sort_values(ascending=False))

#from sklearn.preprocessing import OneHotEncoder,LabelEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# Using the elbow method to find number of clusters
from sklearn.cluster import KMeans
inter_cluster_variance = []
for i in range(1, 11):
    k_means = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    k_means.fit(X)
    inter_cluster_variance.append(k_means.inertia_)
plt.plot(range(1, 11), inter_cluster_variance)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inter Cluster Variance')
plt.show()

# Training the K-Means model after finding optimal number of clusters from elbow method
k_means = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
cluster_group = k_means.fit_predict(X)
print(cluster_group)
dataset['cluster'] = cluster_group
#print(dataset)
dataset.to_csv('after_clustering.csv')

plt.scatter(X[cluster_group == 0, 0], X[cluster_group == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[cluster_group == 1, 0], X[cluster_group == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[cluster_group == 2, 0], X[cluster_group == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[cluster_group == 3, 0], X[cluster_group == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[cluster_group == 4, 0], X[cluster_group == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Salary')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


