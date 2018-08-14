# https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
# KNN with multiple labels
# this calculates Euclidean distance
# but is sensitive to magnitudes;
# data must be scaled accordingly


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

#test size can also be random
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
 test_size=0.3) # 70% training and 30% test

#create the classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

#train the set
knn.fit(X_train, y_train)

# amke the prediction
y_prediction = knn.predict(X_test)
# print(y_prediction)

#judge the accuracy
from sklearn import metrics

print('Accuracy: ', metrics.accuracy_score(y_test, y_prediction))