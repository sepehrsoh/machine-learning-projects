import pandas as pd
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    accuracy = model.score(val_X, val_y)
    return (mae), accuracy


if __name__ == '__main__':
    melbourne_file_path = "melb_data.csv"
    melbourne_data = pd.read_csv(melbourne_file_path)
    print(melbourne_data.columns)
    # The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
    # We'll learn to handle missing values in a later tutorial.
    # Your Iowa data doesn't have missing values in the columns you use.
    # So we will take the simplest option for now, and drop houses from our data.
    # Don't worry about this much for now, though the code is:

    # dropna drops missing values (think of na as "not available")
    melbourne_data = melbourne_data.dropna(axis=0)

    y = melbourne_data.Price
    # things we think are efficient on house price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    # X is columns are important with their data
    X = melbourne_data[melbourne_features]
    # show x
    print(X.describe())
    # show first 5 rows
    print(X.head())

    # split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    # Define model
    melbourne_model = DecisionTreeRegressor()
    # Fit model
    melbourne_model.fit(train_X, train_y)

    # get predicted prices on validation data
    val_predictions = melbourne_model.predict(val_X)
    print(mean_absolute_error(val_y, val_predictions))
    print("test accuracy {}".format(melbourne_model.score(val_X, val_y)))

    # now we find optimal leaf nodes
    # compare MAE with differing values of max_leaf_nodes
    optimal = 0
    mean_abs_error = sys.maxsize
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae, accuracy = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d And accuracy %f" % (max_leaf_nodes, my_mae, accuracy))
        if my_mae < mean_abs_error:
            mean_abs_error = my_mae
            optimal = max_leaf_nodes
    print("optimal max leaf nodes is {} with Mean Absolute Error {}".format(optimal, int(mean_abs_error)))
    print("now trying random forest method")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    melb_preds = forest_model.predict(val_X)
    print(
        "random forest model  Mean Absolute Error = {} and accuracy = {}".format(int(mean_absolute_error(val_y, melb_preds)),
                                                                                 forest_model.score(val_X, val_y)))
