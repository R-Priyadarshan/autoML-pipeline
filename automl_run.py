# automl_run.py
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('iot_classification.csv')
X = df.drop('label', axis=1)
y = df['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

pipeline = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
pipeline.fit(X_train, y_train)
print('score', pipeline.score(X_test,y_test))
pipeline.export('best_pipeline.py')
print('exported best_pipeline.py')
