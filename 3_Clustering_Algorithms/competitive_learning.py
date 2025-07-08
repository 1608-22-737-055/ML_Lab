import pandas as pd
import numpy as np

df = pd.read_csv('../datasets/mall_customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

n_clusters = 5
centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
lr = 0.5

for epoch in range(10):
    for x in X:
        dists = np.linalg.norm(centroids - x, axis=1)
        idx = np.argmin(dists)
        centroids[idx] += lr * (x - centroids[idx])
    lr *= 0.9

labels = np.argmin(np.linalg.norm(centroids[:, None, :] - X, axis=2), axis=0)
df['Cluster'] = labels

print(df[['CustomerID', 'Cluster']].head())
