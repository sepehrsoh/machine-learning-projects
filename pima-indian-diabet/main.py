import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# header's name of data
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# reading data with pandas library from csv file
pima = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)
# show head of data which we read from csv file
print("head of data : \n {}".format(pima.head()))
# column we want select to use it for fitting the model
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]  # Features
y = pima.label  # Target variable
# define train values and test for train and test DecisionTreeClassifier model
# split 30% of data for test and use rest of them fot train model (test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# define model and fit it with train data
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
# get predict with trained model using test data
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# confusion matrix
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n {}".format(result))
# report of classification
report = classification_report(y_test, y_pred)
print("Classification Report: \n {}".format(report))
#
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))
