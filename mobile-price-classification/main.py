import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')

    print(train_df.columns.values)
    print(train_df.head())

    train_df.dropna(axis=0)

    mobile_features = ['battery_power', 'dual_sim', 'ram', 'int_memory', 'sc_h', 'sc_w', 'touch_screen']
    X = train_df[mobile_features]
    print(X.head())

    y = train_df.price_range

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    mobile_model = DecisionTreeRegressor()

    mobile_model.fit(train_X, train_y)
    predictions = mobile_model.predict(val_X)
    print("-" * 40)
    print("test accuracy with Decision Tree Regressor {}".format(mobile_model.score(val_X, val_y)))

