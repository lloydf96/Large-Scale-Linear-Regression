# Large-Scale-Linear-Regression
Implemented a linear regression model to predict taxi fare using the New York taxi dataset from Kaggle which contains data from 20M+ trips.
Used Geo-hashing to split New York city into smaller blocks and converted the location data into categorical variables and augmented the dataset with hourly weather data. 
Data preprocessing and feature engineering were parallelized using Dask. Implemented a parallelized mini-batch Gradient Descent algorithm using Dask achieving an RMSE of 3.4. 
Link for the dataset: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction, https://www.kaggle.com/selfishgene/historical-hourly-weather-data

