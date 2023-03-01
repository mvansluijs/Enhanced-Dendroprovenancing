import os

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from functions.extract_data import get_downscaled_cru_ts
from functions.extract_data import get_soil

# Load overview table
overview = pd.read_csv('overview_after_1_2.csv', sep=';')

# Load points array and convert to list of tuples
points = np.load('prediction_points.npy')
pred_coords = [(x, y) for x, y in zip(points[:, 0], points[:, 1])]

# Get soil- and meteo data
soil = get_soil(pred_coords)
meteo = get_downscaled_cru_ts(pred_coords)

# Create counter for the amount of temporary files created
tempfilecounter = 0

# Loop over the points, for each point create data records like in the training table
for i in tqdm(range(len(points)), desc="Creating prediction table", smoothing=0):
    # Access meteo data, calculate centered rolling mean and deviation from rolling mean and combine into one
    point_meteo = pd.DataFrame(meteo[i])
    point_meteo.index = range(1901, 2022)
    point_meteo_avg = point_meteo.rolling(15, min_periods=1, center=True, axis=0).mean()
    point_meteo_dev = point_meteo - point_meteo_avg
    meteo_full = pd.concat([point_meteo, point_meteo_avg, point_meteo_dev], axis=1)

    # Access soil data and make the same length as the meteo data by duplication (soil data stays the same every year)
    point_soil = pd.DataFrame(soil[i])
    point_soil = pd.concat([point_soil.T] * len(point_meteo))
    point_soil.index = range(1901, 2022)

    # Combine soil- and meteo data
    combined = pd.merge(meteo_full, meteo_full.shift(), left_index=True, right_index=True).dropna()
    combined = pd.merge(combined, point_soil, left_index=True, right_index=True, suffixes=('_i', '_j')).dropna()

    # Add columns for year, latitude and longitude
    combined['year'] = list(combined.index)
    combined['lat'] = points[i][0]
    combined['lon'] = points[i][1]

    # Convert to list of lists, extend if possible, else if total list non existent create
    try:
        total.extend(combined.to_numpy().tolist())
    except NameError:
        total = combined.to_numpy().tolist()

    # Save to a temporary file every 2500 iterations to prevent memory overload issues
    if i % 2500 == 2499:
        # Convert back to numpy array
        total = np.asarray(total)

        # Add zeros in the columns where the training table has one-hot encoding
        zeros = np.zeros((len(total), len(overview)))
        total = np.hstack((total, zeros))

        # Save array to file
        np.save('temp/prediction_table' + str(tempfilecounter) + '.npy', total)
        tempfilecounter += 1

        # Remove 'total' from memory
        del total

# The same operation as the last part within the loop, but now for the remainder
total = np.asarray(total)
zeros = np.zeros((len(total), len(overview)))
total = np.hstack((total, zeros))
np.save('temp/prediction_table_' + str(tempfilecounter) + '.npy', total)
del total
del soil
del meteo

# Recombine the temp files into one big array and remove temporary files
for filename in os.listdir('temp'):
    f = os.path.join('temp', filename)

    try:
        prediction_table = np.vstack((prediction_table, np.load(f)))
    except NameError:
        prediction_table = np.load(f)

    os.remove(f)

np.save('prediction_table.npy', prediction_table)

# Iterate over the records in the overview table (chronologies)
for i, record in tqdm(overview.iterrows(), desc="Creating modelled chronologies", total=len(overview)):
    # Calculate the index of the record in the overview data (counts front-to-back)
    record_index = len(overview['name']) - i

    # Load the random forest model to be used for the current record
    rf = joblib.load('2_1_random_forests/rf_without_' + str(record['name']) + '.joblib')

    # Use the random forest model to predict the values in the prediction table
    modelled_crn = rf.predict(prediction_table)
    modelled_crn = modelled_crn.reshape(int(len(prediction_table) / 120), 120)

    # Save array to file
    np.save('2_2_modelled_chronologies/modelled_crn_without_' + str(record['name']) + '.npy', modelled_crn)
