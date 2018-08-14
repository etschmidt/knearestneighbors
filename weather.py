# X variables that would determine outcomes
# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Y variable
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

# Import LabelEncoder
from sklearn import preprocessing

# create the encoder
le = preprocessing.LabelEncoder()

weather_encoded = le.fit_transform(weather)
temp_encoded = le.fit_transform(temp)
play_encoded = le.fit_transform(play)

# print(weather_encoded, temp_encoded, play_encoded)

#combine the data into single set
#this makes a Euclidean plane
features = list(zip(weather_encoded, temp_encoded))

from sklearn.neighbors import KNeighborsClassifier

#use the model and set initial KNN
model = KNeighborsClassifier(n_neighbors=3)

#train the model
model.fit(features, play)

#predict using 0 as overcast and 2 for mild
prediction = model.predict([[0,2]])

print(prediction) #should be 'Yes'

