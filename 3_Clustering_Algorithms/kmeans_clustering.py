import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../datasets/mall_customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

X_scaled = StandardScaler().fit_transform(X)

model = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = model.fit_predict(X_scaled)

print(df[['CustomerID', 'Cluster']].head())
