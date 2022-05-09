from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import argparse


# load data
dataset = loadtxt('con_data.csv', delimiter=",")
# split data into X and y
n = len(dataset)
m= len(dataset[0])
print(n,m)
X = dataset[:,0:m-2]
Y = dataset[:,m-1]
# split data into train and test sets
seed = 7
test_size = 0.20
print("build train and test")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
# print("Fitting XGB model")
# model.fit(X_train, y_train)
# make predictions for test data
model.load_model("model.json")

# model.save_model("model.json")

print("Predicting")
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
F1 = f1_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("F1: %.2f%%" % (F1 * 100.0))