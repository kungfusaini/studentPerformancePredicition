import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import joblib

data = pd.read_csv("student-mat.csv", sep=";")

# List of attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
data = shuffle(data)

# The label, aka what we wish to predict
predict = "G3"

# The data without what we wish to predict
# and what we wish to predict
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Create test and train data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Train the model
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# Dump the model
joblib.dump(linear, 'model.pkl')
print("Model dumped!")

# Load the model
linear = joblib.load('model.pkl')

# Save the data columns
model_columns = list(data.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

# accuracy = linear.score(x_test,y_test)
#
#
# print("Accuracy: ", accuracy)
#
# predictions = linear.predict(x_test)
#
# for x in range(len(x_test)):
#     print(predictions[x], x_test[x], y_test[x])
#
#
# sample = [[12, 13, 0, 0, 0]]
# predictions = linear.predict(sample)
#
# print(predictions, sample)
#
