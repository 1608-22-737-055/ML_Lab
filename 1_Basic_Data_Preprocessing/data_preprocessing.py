import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('../datasets/iris.csv')
print("Original data sample:\n", df.head())

# Encode labels
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Standardize features
scaler = StandardScaler()
features = df.columns[:-1]
df[features] = scaler.fit_transform(df[features])

print("\nPreprocessed data sample:\n", df.head())
