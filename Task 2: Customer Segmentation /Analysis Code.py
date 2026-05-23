# Customer Segmentation Using ML - Dataset: Mall Customer.csv

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Dataset 
data = pd.read_csv('2. Customer Segmentation Dataset.csv')

# Display Dataset
print("\nFirst 5 Rows:\n", data.head())
print("\nDataset Shape:\n",data.shape)
print("\nColumn Names:\n",data.columns)
print("\nDataset Information:\n",data.info())
print("\nMissing Values:\n",data.isnull().sum())

# Basic Data Analyis
print("\nStatistical Summary:\n",data.describe())

# Data Visualization Part

# 1. Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 2. Gender Count
plt.figure(figsize=(5,4))
sns.countplot(x='Gender', data=data)
plt.title('Gender Distribution')
plt.show()

# 3. Annual Income Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Annual Income (k$)'], bins=20, kde=True)
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Count')
plt.show()

# 4. Spending Score Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Spending Score (1-100)'], bins=20, kde=True)
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.show()

# 5. Scatter Plot
plt.figure(figsize=(7,5))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=data)
plt.title('Income vs Spending Score')
plt.show()

# Feature Selection: Selecting important columns for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print("\nSelected Features:\n",X.head())

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal No. of Clusters & Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means Clustering:Elbow Graph Choose 5 Clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters
print("\nClustered Data:\n",data.head())

# Silhouette Score
score = silhouette_score(X_scaled, clusters)
print("\nSilhouette Score:\n",score)

# Visualize Customer Segments
plt.figure(figsize=(8,6))
sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette='Set1', s=100)

# Cluster Centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:,0], centroids[:,1], s=300, c='black', marker='X', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Cluster Analysis
print("\n*** Cluster Analysis ***")
cluster_summary = data.groupby('Cluster').mean(numeric_only=True)
print(cluster_summary)

# Customer Segement Interpretation
print("\n*** Customer Segments ***")

for cluster in sorted(data['Cluster'].unique()):

    print(f"\nCluster {cluster}")

    avg_income = data[data['Cluster']==cluster]['Annual Income (k$)'].mean()
    avg_spending = data[data['Cluster']==cluster]['Spending Score (1-100)'].mean()

    print("Average Income:", round(avg_income,2))
    print("Average Spending Score:", round(avg_spending,2))

    if avg_income > 70 and avg_spending > 70:
        print("Customer Type: High Income - High Spending")

    elif avg_income > 70 and avg_spending < 40:
        print("Customer Type: High Income - Low Spending")

    elif avg_income < 40 and avg_spending > 60:
        print("Customer Type: Low Income - High Spending")

    elif avg_income < 40 and avg_spending < 40:
        print("Customer Type: Low Income - Low Spending")

    else:
        print("Customer Type: Moderate Customers")

# Pairplot Visualization
sns.pairplot(data, hue='Cluster')
plt.show()

# Save Output File
output_file = '2. Customer Segmentation Result.csv'
data.to_csv(output_file, index=False)
print("\nResult Saved")
print(f"Clustered dataset saved as: {output_file}")

# Project Completed
print("\n*** Project Completed Successfully ***")
