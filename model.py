import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

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

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test,y_test)
print("Accuracy: ", accuracy)

predictions = linear.predict(x_test)

for x in range(len(x_test)):
    print(predictions[x], x_test[x], y_test[x])


sample = [[12, 13, 0, 0, 0]]
predictions = linear.predict(sample)

print(predictions, sample)

