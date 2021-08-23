import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

# List of attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# The label, aka what we wish to predict
predict = "G3"

# The data without what we wish to predict
# and what we wish to predict
x = np.array(data.drop([predict]), 1)
y = np.array(data[predict])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
