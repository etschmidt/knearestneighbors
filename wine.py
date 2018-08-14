# https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
# KNN with multiple labels
# get the wine data set

from sklearn import datasets

wine = datasets.load_wine()

''' Just  to visualize better
import pandas as pd

df = pd.DataFrame(columns=wine.feature_names, data=wine.data)

df['target'] = wine.target
print(df)
'''

# print(wine.feature_names)
# print(wine.target_names)

# print the top 5
# print(wine.data[0:5])
# print(wine.target)

# split the data into train/test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test
