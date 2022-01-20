import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    # path for file we are reading data from it
    train_path = "train.csv"
    # with pandas library reading csv file witch stored data in it
    train_data = pd.read_csv(train_path)
    # y is the column have answers
    y = train_data.SalePrice

    print("Headers of data : {} ".format(train_data.columns.values))

    # Create the list of features below
    # These features selected as important thing that mainly effects on our result
    feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

    # select data corresponding to features in feature_names
    X = train_data[feature_names]

    # split data for train model and check accuracy
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # define model to use an algorithm to fit model
    model = RandomForestRegressor()
    # Fit the model
    model.fit(train_X, train_y)
    # predict data with trained model
    predictions = model.predict(val_X)
    print("test accuracy with Random Forest Regressor {} %".format(model.score(val_X, val_y) * 100))
    print("____________________________________")
    print("train accuracy with Random Forest Regressor {} %".format(model.score(train_X, train_y) * 100))

