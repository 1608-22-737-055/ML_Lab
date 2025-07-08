import pandas as pd
from minisom import MiniSom

df = pd.read_csv('../datasets/iris.csv')
X = df.drop('species', axis=1).values

som = MiniSom(x=7, y=7, input_len=4, sigma=1.0, learning_rate=0.5, random_seed=42)
som.train_random(X, 100)

for i in range(10):
    print(f"Sample {i}: BMU -> {som.winner(X[i])}")
