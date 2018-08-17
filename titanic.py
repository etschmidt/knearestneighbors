# In Machine Learning, the types of Learning can broadly be classified 
# into three types: 1. Supervised Learning, 2. Unsupervised Learning and 
# 3. Semi-supervised Learning. Algorithms belonging to the family of 
# Unsupervised Learning have no variable to predict tied to the data. 
# Instead of having an output, the data only has an input which would be 
# multiple variables that describe the data. This is where clustering comes in.

# Cluster Titanic passengers into Survival categories based on their data

# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

train_data = "data/train.csv"
train = pd.read_csv(train_data)
test_data = "data/test.csv"
test = pd.read_csv(test_data)

# print("***** Train_Set *****")
# print(train.describe())
# print("\n")
# print("***** Test_Set *****")
# print(test.describe())

# find NA values to be filed in 
# print(train.isna().head())
# print(train.isna().sum())

# Now, there are several ways you can fill in NAs:

# A constant value that has meaning within the domain, such as 0, distinct from all other values.
# A value from another randomly selected record.
# A mean, median or mode value for the column.
# A value estimated by another machine learning model.

# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(), inplace=True)

# print(test.isna().sum())

# Let's do some more analytics in order to understand the data better. 
# Understanding is really required in order to perform any Machine Learning task. 
# Let's start with finding out which features are categorical and which are numerical.

# Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
# Continuous: Age, Fare. Discrete: SibSp, Parch.

# print(train['Ticket'].head())

# Find the Survival count wrt Pclass:
# print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False)
# 	.mean().sort_values(by='Survived', ascending=False))

# # and by sex:
# print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False)
# 	.mean().sort_values(by='Survived', ascending=False))
# # NUmber so siblings or spouses
# print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False)
# 	.mean().sort_values(by='Survived', ascending=False))

# plot the ages vs survival:
# g = sns.FacetGrid(train, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# Its time to see how the Pclass and Survived features are related to eachother with a graph:
# grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();
# plt.show()

# print(train.info())
# Name, sex, ticket, cabin, and embarked are non-numberic and must be converted
# or removed; those that don't impact survival should be removed

train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# now encode sex to numerical values

le = LabelEncoder()
le.fit(train['Sex'])
le.fit(test['Sex'])
train['Sex'] = le.transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

# print(train.info())

# build the model without using the 'Survival' column of the train data
X = np.array(train.drop(['Survived'], 1).astype(float))
# and then use it for the y values
y = np.array(train['Survived'])

# print(X)

# Now we have good, clean, pure data
# and can build out model
kmeans = KMeans(n_clusters=2) #obviously, there are two clusters
kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(X)

# K means is greatly affected by discrepencies in magnitude
# take 0 - 1 as the uniform value range across all the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)

# print(kmeans.fit(X))

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

# Let's see how well the model is doing by looking at the percentage of 
# passenger records that were clustered correctly.
print(correct/len(X))

# The biggest disadvantage is that K-Means requires you to pre-specify the number 
# of clusters (k). However, for the Titanic dataset, you had some domain knowledge 
# available that told you the number of people who survived in the shipwreck. 
# This might not always be the case with real world datasets. Hierarchical clustering
# is an alternative approach that does not require a particular choice of clusters. 
# An additional disadvantage of k-means is that it is sensitive to outliers and 
# different results can occur if you change the ordering of the data.

# K-Means is a lazy learner where generalization of the training data is delayed 
# until a query is made to the system. This means K-Means starts working only when
# you trigger it to, thus lazy learning methods can construct a different approximation 
# or result to the target function for each encountered query. It is a good method 
# for online learning, but it requires a possibly large amount of memory to store the 
# data, and each request involves starting the identification of a local model from 
# scratch.