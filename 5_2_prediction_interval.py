from sklearn_quantile import RandomForestQuantileRegressor
import numpy as np
import pandas as pd

# Load training- and prediction table
training_table = np.load('training_table_sigma_4.0.npy')
prediction_table = np.load('prediction_table.npy')

# Train quantile regression forest (QRF)
qrf = RandomForestQuantileRegressor(n_jobs=-1, n_estimators=200, random_state=42, q=[0.05, 0.95])
qrf.fit(training_table[:, 1:], training_table[:, 0])

# Predict prediction table
qrf_out = qrf.predict(prediction_table)

# Extract prediction interval
prediction_interval = qrf_out[1] - qrf_out[0]

# Get longitudes and latitudes
lon = prediction_table[:, -109]
lat = prediction_table[:, -108]

# Couple all points to their corresponding prediction interval and average per point
df = pd.DataFrame({'lon': lon, 'lat': lat, 'prediction_interval': prediction_interval})
df_grouped = df.groupby(['lon', 'lat']).mean()

# Export to csv
df_grouped.to_csv('4_0_general_outputs/prediction_interval.csv')
