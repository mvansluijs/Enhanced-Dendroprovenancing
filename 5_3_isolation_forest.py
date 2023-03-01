from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

# Load training table and prediction table
training_table = np.load('training_table_sigma_4.0.npy')
prediction_table = np.load('prediction_table.npy')

# Train isolation forest
clf = IsolationForest(n_jobs=-1, n_estimators=200, random_state=42)
clf.fit(training_table[:, 1:])

# Calculate path length of prediction
path_length = clf.predict(prediction_table)

# Get longitudes and latitudes
lon = prediction_table[:, -109]
lat = prediction_table[:, -108]

# Couple all points to their corresponding isolation forest path length and average per point
df = pd.DataFrame({'lon': lon, 'lat': lat, 'path_length': path_length})
df_grouped = df.groupby(['lon', 'lat']).mean()

# Export to csv
df_grouped.to_csv('4_0_general_outputs/isolation_forest.csv')
