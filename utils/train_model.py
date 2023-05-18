"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed', 'Bilbao_rain_1h', 'Valencia_wind_speed']]

# Define the base models
random_forest = RandomForestRegressor()
decision_tree = DecisionTreeRegressor()
gradient_boosting = GradientBoostingRegressor()
xgboost = XGBRegressor()
adaboost = AdaBoostRegressor()
svr = SVR()

# Define the stacked ensemble model
estimators = [
    ('random_forest', random_forest),
    ('decision_tree', decision_tree),
    ('gradient_boosting', gradient_boosting),
    ('xgboost', xgboost),
    ('adaboost', adaboost),
    ('svr', svr)
]

stacked_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(normalize=True)
)

# Fit the stacked ensemble model
print("Training Model...")
stacked_model.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/stacked_ensemble_regression.pkl'
print(f"Training completed. Saving model to: {save_path}")
pickle.dump(stacked_model, open(save_path, 'wb'))

