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

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

print(train.columns)
print(test.columns)