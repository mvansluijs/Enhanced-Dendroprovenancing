import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from functions.extract_data import get_downscaled_cru_ts
from functions.extract_data import get_soil

# Load the overview table
overview = pd.read_csv('overview_after_1_2.csv', sep=';')

# Create a list of tuples containing the longitude and latitude values
pred_coords = [(x, y) for x, y in zip(overview.longitude, overview.latitude)]

# Get soil and meteo data
soil = get_soil(pred_coords)
meteo = get_downscaled_cru_ts(pred_coords)

# Initiate list to be filled with R squared values
rsq_list = []

# Loop over the chronologies in the overview table
for i, record in tqdm(overview.iterrows(), desc="Calculating...", smoothing=0):
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
    combined['lat'] = record.latitude
    combined['lon'] = record.longitude

    # Add zeros in the columns where the training table has one-hot encoding
    zeros = np.zeros((len(combined), len(overview)))
    combined = np.hstack((combined, zeros))

    # Get the chronology data and split into target years and target (chronology) values
    in_path = '1_3_chronologies/' + record['name'] + '_sigma_' + str(record['use_sigma']) + '_crn.csv'
    target_chronology = np.genfromtxt(in_path, delimiter=';')
    target_years = target_chronology[:, 0]
    target_values = target_chronology[:, 1]

    # Load the random forest for this chronology
    rf = joblib.load('2_1_random_forests/rf_without_' + str(record['name']) + '.joblib')

    # Model a chronology based on the conditions of its site
    out = rf.predict(combined)
    out = out.reshape(int(len(out) / 120), 120)

    # Calculate Pearson R for offsets from -5 to +5
    rsq = []
    for j in range(-5, 6):
        subset_years = np.in1d(np.arange(1902, 2022), target_years + j)
        subset_years2 = np.in1d(target_years + j, np.arange(1902, 2022))

        rsq.append(np.corrcoef(out[0][subset_years], target_values[subset_years2])[0, 1])
    rsq_list.append(rsq)

# Convert list of lists to Pandas dataframe
df = pd.DataFrame(rsq_list)

# Change to 0 where Pearson R < 0, as the slope is negative there and square to obtain R squared
df[df < 0] = 0
df = df ** 2

# Save raw results to file
df.to_csv('4_0_general_outputs/year_unknown.csv', index=False, header=False, sep=';')

# Add the offset with the highest r squared value to the overview table
overview['year_known_offset'] = np.argmax(df.to_numpy(), axis=1) - 5

# Write new overview table
overview.to_csv('overview_after_4_1.csv', sep=';', index=False)
