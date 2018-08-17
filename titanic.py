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

print(test.isna().sum())