# KNN
### Knn - K nearest nerherbour

#### Steps 

> Meemorizes Trainig dataset

> Calculates distance between testing data point and all other traing data points

> Based on K value its gona pick K neares nighbours 

> Based on the majority its predict the class (YES/NO)

NOTE: Pick K has odd number
  
- Its called as lazy learning algorithm
- ### Knn - K nearest nerherbour

#### Steps 

> Meemorizes Trainig dataset

> Calculates distance between testing data point and all other traing data points

> Based on K value its gona pick K neares nighbours 

> Based on the majority its predict the class (YES/NO)

NOTE: Pick K has odd number
  
- Its called as lazy learning algorithm

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as knn

df = load_iris()
df

dfi = pd.DataFrame(df.data, columns=df.feature_names)
dfi.head()

knn = knn(n_neighbors=3)
y_label = df.target
dfi

y_label
features = dfi.iloc[:, 0:4].values
features

x_train, x_test, y_train, y_test = train_test_split(features, y_label, test_size=0.2)
