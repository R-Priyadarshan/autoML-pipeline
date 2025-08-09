# generate_dataset.py
import numpy as np
import pandas as pd

N = 2000
np.random.seed(0)
X1 = np.random.normal(0,1,(N,3))
X2 = np.random.normal(5,1,(N,2))
X = np.hstack([X1,X2])
labels = (X[:,0] + X[:,3] > 4).astype(int)
df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
df['label'] = labels
df.to_csv('iot_classification.csv', index=False)
print('saved iot_classification.csv')
