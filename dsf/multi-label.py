import numpy as np

import DTE


from DTE import DTE
import ReadData
from sklearn.metrics import accuracy_score
import pandas as pd

dataset = 'magic'
dataPath = "../data/"

X_train, Y_train = ReadData.readData(dataset, 'train', dataPath)
X_test, Y_test = ReadData.readData(dataset, 'test', dataPath)

X_train = pd.DataFrame.from_records(X_train)
X_test = pd.DataFrame.from_records(X_test)
#Y_train = np.array(Y_train)
print(type(X_train))
features = True
output_space_features = False
tree_embedding_features = True

df = DTE(task = "mlc", features = features, output_space_features = output_space_features, tree_embedding_features = tree_embedding_features)
df.fit(X_train,Y_train,X_test,Y_test)
predictions = df.predict(X_train)

accuracy = accuracy_score(Y_test, predictions)
print(accuracy)