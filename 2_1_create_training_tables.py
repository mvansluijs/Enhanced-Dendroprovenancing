import pandas as pd
import numpy as np

from functions.extract_data import get_soil
from functions.extract_data import get_downscaled_cru_ts

# Load the overview table
overview = pd.read_csv('overview_after_1_2.csv', sep=';')

# Create a list of tuples containing the longitude and latitude values
train_coords = [(x, y) for x, y in zip(overview.longitude, overview.latitude)]

# Get soil and meteo data
soil = get_soil(train_coords)
meteo = get_downscaled_cru_ts(train_coords)

# Loop over all possible sigma's that will be used later in the analysis
for sigma in overview['use_sigma'].unique():
    # Variable management
    try:
        del total
    except NameError:
        pass

    # Loop over chronologies
    for i in range(len(overview)):
        record = overview.iloc[i]

        # Extract name to be used for current chronology
        code = record['name']

        # Specify chronology path and load chronology
        in_path = '1_3_chronologies/' + str(code) + '_sigma_' + str(sigma) + '_crn.csv'
        crn = pd.read_csv(in_path, sep=';', index_col=0, header=None)

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

        # Combine chronology, soil- and meteo data, dropping years that are not found in all datasets
        combined = pd.merge(crn, meteo_full, left_index=True, right_index=True).dropna()
        combined = pd.merge(combined, meteo_full.shift(), left_index=True, right_index=True).dropna()
        combined = pd.merge(combined, point_soil, left_index=True, right_index=True, suffixes=('_i', '_j')).dropna()

        # Add columns for year, latitude, longitude and site number
        combined['year'] = list(combined.index)
        combined['lat'] = record['latitude']
        combined['lon'] = record['longitude']
        combined['site_nr'] = i

        # Concatenate to the other chronologies ('total'), if it is the first, define 'total'
        try:
            total = pd.concat((total, combined), axis=0)
        except NameError:
            total = combined

    # Create one-hot encoding using the site number
    one_hot = pd.get_dummies(total['site_nr'].astype(str).str.zfill(5))

    # Remove the site number
    total = total.iloc[:, :-1]

    # Concatenate the one-hot encoding to the total training table
    total = pd.concat([total, one_hot], axis=1)

    # Save table to file
    np.save('training_table_sigma_' + str(sigma) + '.npy', total)
