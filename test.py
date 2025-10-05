import pandas as pd
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['label'] = y
df.to_csv('C:/Users/Ammar1/Downloads/moons.csv', index=False)

