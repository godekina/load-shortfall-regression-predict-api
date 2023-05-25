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
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
       'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
       'Valencia_pressure', 'Seville_temp_max', 'Bilbao_weather_id', 
        'Valencia_humidity', 'Year', 'Month_of_year', 'Day_of_month', 'Day_of_week', 'Hour_of_day']]

# Fit models
RF = RandomForestRegressor(n_estimators=150, min_samples_split=2, min_samples_leaf=1, max_depth=110, bootstrap=True)
RF.fit(X_train,y_train)

krr = KernelRidge(kernel='laplacian',alpha=0.01)
krr.fit(X_train, y_train)

# Pickle model of RF for use within our API
save_path = '../assets/trained-models/RF.pkl'
print(f"Training completed. Saving model to: {save_path}")
pickle.dump(RF, open(save_path, 'wb'))

# Pickle model of KRR for use within our API
save_path = '../assets/trained-models/krr.pkl'
print(f"Training completed. Saving model to: {save_path}")
pickle.dump(krr, open(save_path, 'wb'))
