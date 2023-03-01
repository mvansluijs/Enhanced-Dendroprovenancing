import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn_quantile import RandomForestQuantileRegressor

from functions.extract_data import get_downscaled_cru_ts
from functions.extract_data import get_soil

# Load overview- and training table
overview = pd.read_csv('overview_after_4_3.csv', sep=';')

# State index of interest and select corresponding record from overview table
index = -42
record = overview.iloc[index]

# Get the correct training table for the selected record/index
training_table = np.load('training_table_sigma_' + str(record['use_sigma']) + '.npy')

# Get the latitude and longitude
lat = record[1]
lon = record[2]

# Set coords into a list of tuples for data extraction
pred_coords = [(lon, lat)]

# Extract soil- and meteo data
soil = get_soil(pred_coords)
meteo = get_downscaled_cru_ts(pred_coords)

# Set parameters for creating a prediction table
train_n = 107
roll = 15

# Create a prediction table for this site
point_meteo = pd.DataFrame(meteo[0])
point_meteo.index = range(1901, 2022)
point_meteo_avg = point_meteo.rolling(roll, min_periods=1, center=True, axis=0).mean()
point_meteo_dev = point_meteo - point_meteo_avg

meteo_full = pd.concat([point_meteo, point_meteo_avg, point_meteo_dev], axis=1)

point_soil = pd.DataFrame(soil[0])
point_soil = pd.concat([point_soil.T] * len(point_meteo))
point_soil.index = range(1901, 2022)

combined = pd.merge(meteo_full, meteo_full.shift(), left_index=True, right_index=True).dropna()
combined = pd.merge(combined, point_soil, left_index=True, right_index=True, suffixes=('_i', '_j')).dropna()

combined['year'] = list(combined.index)
combined['lat'] = lat
combined['lon'] = lon

zeros = np.zeros((len(combined), train_n))
combined = np.hstack((combined, zeros))

# Train a quantile RF regressor and a normal RF regressor based on the training table
train = training_table[(training_table[:, index] != 1)]
test = training_table[(training_table[:, index] == 1)]

train_inputs = np.delete(train, 0, 1)
train_targets = train[:, 0]

qrf = RandomForestQuantileRegressor(n_estimators=500, n_jobs=-1, random_state=42, q=[0.10, 0.90])
qrf.fit(train_inputs, train_targets)

rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
rf.fit(train_inputs, train_targets)

# Predict the prediction table
test_inputs = combined
test_targets = test[:, 0]

y_pred_qrf = qrf.predict(test_inputs)
y_pred_rf = rf.predict(test_inputs)

# Rescale the results
y_pred_qrf[0] = (y_pred_qrf[0] - np.mean(y_pred_rf)) / np.std(y_pred_rf)
y_pred_qrf[1] = (y_pred_qrf[1] - np.mean(y_pred_rf)) / np.std(y_pred_rf)
y_pred_rf = (y_pred_rf - np.mean(y_pred_rf)) / np.std(y_pred_rf)
test_targets = (test_targets - np.mean(test_targets)) / np.std(test_targets)

# Plotting
y_lower = y_pred_qrf[0]
y_mean = y_pred_rf
y_upper = y_pred_qrf[1]

xx = np.array(range(len(y_mean)))
xx += 1902

xx2 = training_table[(training_table[:, index] == 1)][:, -110]

fig = plt.figure(figsize=(10, 10))
plt.plot(xx, y_mean, 'r-', label='Prediction')
plt.plot(xx2, test_targets, 'g-', label='Ground truth')
plt.plot(xx, y_upper, 'k-')
plt.plot(xx, y_lower, 'k-')
plt.fill_between(xx.ravel(), y_lower, y_upper, alpha=0.4,
                 label='80% prediction interval')
plt.ylim(np.min(y_lower) - .2, np.max(y_upper) + .2)
plt.legend(loc='upper left')
plt.show()
