import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


data = pd.read_csv("data-set.csv")
print(data.describe())
print(data.info())

useful_features = ["Sex", "Age", "Pclass"]

y = data.Survived  # Target 1=Survived, 0=Did Not Survive
test = data.drop(columns=["PassengerId"], axis=1).copy()
X = data[useful_features]

# some age data are missing here is the number
n = X["Age"].isna().sum()
print(f"Number missing: {n}")

# best way to fill data is using mean value to do not change result
m = X["Age"].mean()
print("mean value of age", m)
# do not count null data in calculate median and fill missing data
data["Age"].fillna(data["Age"].median(skipna=True), inplace=True)
data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(), inplace=True)

impute_mean = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=1)
m = impute_mean.fit_transform(X[["Age"]])
mt = impute_mean.transform(test[["Age"]])

X["Age"] = impute_mean.fit_transform(X[["Age"]])
test["Age"] = impute_mean.transform(test[["Age"]])
# Verify no more Age values with na
n = X["Age"].isna().sum()
print(f"Number missing: {n}")

# change sex label using number ,for male use 0 for female using 1
X["Sex"] = X["Sex"].replace("male", 0)  # .astype(int)
X["Sex"] = X["Sex"].replace("female", 1)  # .astype(int)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = LogisticRegression(C=0.12, solver="liblinear")
model.fit(train_X, train_y)

predict = model.predict(val_X)

score = accuracy_score(val_y, predict)
model_score = model.score(train_X, train_y)

print("accuracy of prediction : ", score)
print("model score ", model_score)
