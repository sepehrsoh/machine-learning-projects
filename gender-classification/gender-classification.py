import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB


# this function has been define for return accuracy for any model
def clf_scores(clf, y_predicted):
    # Accuracy
    acc_train = clf.score(X_train, y_train) * 100
    acc_test = clf.score(X_test, y_test) * 100

    roc = roc_auc_score(y_test, y_predicted) * 100
    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
    cm = confusion_matrix(y_test, y_predicted)
    correct = tp + tn
    incorrect = fp + fn

    return acc_train, acc_test, roc, correct, incorrect, cm


if __name__ == '__main__':
    dataframe = pd.read_csv("data.csv")
    print("data info : \n {}".format(dataframe.info()))

    # rename columns for better encoding in feature
    dataframe.rename(columns={'Favorite Color': 'FavoriteColor', 'Favorite Music Genre': 'FavoriteMusicGenre',
                              'Favorite Beverage': 'FavoriteBeverage', 'Favorite Soft Drink': 'FavoriteSoftDrink'},
                     inplace=True)

    # create object from columns list
    objFeatures = dataframe.select_dtypes(include="object").columns
    le = preprocessing.LabelEncoder()
    for feat in objFeatures:
        dataframe[feat] = le.fit_transform(dataframe[feat].astype(str))

    print("changed data info : \n {}".format(dataframe.info()))

    # split inputs and result from each other
    X = dataframe.drop(['Gender'], axis=1)
    y = dataframe.Gender

    # create data for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # test data with GaussianNB

    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train, y_train)

    Y_pred_gnc = clf_gnb.predict(X_test)
    results = clf_scores(clf_gnb, Y_pred_gnc)
    print(
        " accuracy train : {} \n "
        "accuracy test : {} \n"
        "roc : {} \n "
        "correct : {} \n "
        "incorrect : {} \n "
        "confusion_matrix: {}".format(
            results[0], results[1], results[2], results[3], results[4], results[5]))
