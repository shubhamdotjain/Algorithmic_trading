

import csv

import numpy

import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets.mldata import fetch_mldata

# Read in the data.
games = pandas.read_csv("Oracle1.csv") 
#Can change the file name above

# Print the names of the columns in games.
print(games.columns)
kmeans_model = KMeans(n_clusters=5, random_state=1)
# Get only the numeric columns from games.
good_columns = games._get_numeric_data()
# Fit the model using the good columns.
kmeans_model.fit(good_columns)
# Get the cluster assignments.
labels = kmeans_model.labels_

pca_2 = PCA(2)

# Fit the PCA model on the numeric columns from earlier.

# Make a scatter plot of each game, shaded according to cluster assignment.
# plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
# Show the plot.


columns = games.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["Close", "Date"]]

# Store the variable we'll be predicting on.
target = "Close"


train=games[:-100]
test=games[-100:]
arr=[]
seed=7
for value in test[target]:

	arr.append(value)
arr1=[]
for value in test['Date']:

	
	arr1.append(value)

actual=arr
# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns], train[target])


predictions = model.predict(test[columns])

# Compute error between our test predictions and the actual values.
root=mean_squared_error(predictions, test[target])

print root
print arr1
print actual
print predictions

plt.plot(predictions, 'red')
plt.plot(actual,'green')
plt.show()


from sklearn.ensemble import RandomForestRegressor

# Initialize the model with some parameters.
model1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model1.fit(train[columns], train[target])
# Make predictions.
predictions1 = model1.predict(test[columns])
# print actual
# print predictions
# Compute the error.
mean_squared_error(predictions, test[target])

