import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

if __name__ == '__main__':
    # read data from csv file
    df = pd.read_csv("heart.csv")
    # show 5 first rows
    print(df.head())
    # target 1 show the Patient has disease and 0 means has not
    countNoDisease = len(df[df.target == 0])
    countHaveDisease = len(df[df.target == 1])
    print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target)) * 100)))
    print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target)) * 100)))
    # show how gender can be effect
    countFemale = len(df[df.sex == 0])
    countMale = len(df[df.sex == 1])
    print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex)) * 100)))
    print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex)) * 100)))
    y = df.target
    x = df.drop(['target'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    accuracies = {}

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    acc = lr.score(x_test, y_test) * 100

    accuracies['Logistic Regression'] = acc
    print("Test Accuracy {:.2f}%".format(acc))

    svm = SVC(random_state=1)
    svm.fit(x_train, y_train)

    acc = svm.score(x_test, y_test) * 100
    accuracies['SVM'] = acc
    print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)

    acc = dtc.score(x_test, y_test) * 100
    accuracies['Decision Tree'] = acc
    print("Decision Tree Test Accuracy {:.2f}%".format(acc))

    rf = RandomForestClassifier(n_estimators=1000, random_state=1)
    rf.fit(x_train, y_train)

    acc = rf.score(x_test, y_test) * 100
    accuracies['Random Forest'] = acc
    print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))

    print("all accuracies : \n {}".format(accuracies))
