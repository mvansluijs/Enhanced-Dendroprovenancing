import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric
from tqdm import tqdm

# Load the overview table
overview = pd.read_csv('overview_after_4_1.csv', sep=';')

# Load the point locations of the modelled chronologies
points = np.load('prediction_points.npy')

# Initiate a list for storing the distances from ground truth to the coordinate with the highest r squared value
distances = []

# Iterate over the records in the overview table
for i, record in tqdm(overview.iterrows(), desc="Calculating...", smoothing=0):
    # Load modelled chronologies (that were fully calculated/generated without using the target chronology)
    modelled_chronologies = np.load('2_2_modelled_chronologies/modelled_crn_without_' + record['name'] + '.npy')

    # Extract the longitudes and latitudes of the modelled chronologies using the points table (point grid)
    lon = points[:, 0]
    lat = points[:, 1]

    # Load the target chronology
    in_path = '1_3_chronologies/' + record['name'] + '_sigma_' + str(record['use_sigma']) + '_crn.csv'
    target_chronology = np.genfromtxt(in_path, delimiter=';')
    target_years = target_chronology[:, 0]
    target_values = target_chronology[:, 1]

    # Calculate correlations between the overlapping years of the target chronology and all modelled chronologies
    subset_years = np.in1d(np.arange(1902, 2022), target_years)
    subset_years2 = np.in1d(target_years, np.arange(1902, 2022))

    pearson_r = 1 - cdist(target_values[subset_years2][np.newaxis],
                          modelled_chronologies[:, subset_years], metric='correlation')[0]

    # Create a data frame with the longitude, latitude, and Pearson R values, and sort it by Pearson R value
    df = pd.DataFrame({'lon': lon, 'lat': lat, 'pearson_r': pearson_r})
    df.sort_values(by=['pearson_r'], ascending=False, inplace=True)

    # Set Pearson R < 0 to 0, excluding negative correlations, and calculate R squared
    df.loc[df.pearson_r < 0, 'pearson_r'] = 0
    df.pearson_r = df.pearson_r ** 2
    df.rename(columns={'pearson_r': 'rsq'}, inplace=True)

    # Save the data frame to a CSV file for visualisation purposes (optional)
    df.to_csv('3_1_year_known_out/' + record['name'] + '_year_known.csv', index=False, sep=';')

    # Get the ground truth longitude and latitude of the current record
    lat1 = record['latitude']
    lon1 = record['longitude']

    # Get the longitude and latitude of the highest Pearson R
    lat2 = df.iloc[0, 1]
    lon2 = df.iloc[0, 0]

    # Convert the longitude and latitude values to radians
    X = [[math.radians(lat1), math.radians(lon1)], [math.radians(lat2), math.radians(lon2)]]

    # Calculate the Haversine distance between the two points (predicted location and ground truth)
    distance_sklearn = 6365 * DistanceMetric.get_metric('haversine').pairwise(X)

    # Print the distance
    distances.append(np.array(distance_sklearn).item(1))

# Add distances to highest r squared coordinate to overview table
overview['location_unknown_distance'] = distances

# Write new overview table
overview.to_csv('overview_after_4_2.csv', sep=';', index=False)

