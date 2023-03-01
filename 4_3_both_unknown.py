import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm


# A fast vectorized numpy implementation of the haversine distance calculation. All inputs must be 1d arrays.
# See https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas/29546836#29546836
def vectorized_haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = np.sin(delta_lat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2.0)**2

    return 6365 * 2 * np.arcsin(np.sqrt(a))


# Load the overview table
overview = pd.read_csv('overview_after_4_2.csv', sep=';')

# Load the point locations of the modelled chronologies
points = np.load('prediction_points.npy')

# Set the maximum comparison length to be used for R squared calculations
max_len = 200

# Initiate lists for storing the results (whether it was in top 1/top 5 or outside top 5 and the r squared of the top
# result)
top_results = []
rsq_top1 = []

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

    # Initiate emtpy lists for the result variables
    pearson_r = []
    offset = []
    lat_new = []
    lon_new = []

    # Loop through the offset range
    for j in range(-5, 6):
        # Get overlapping years
        subset_years = np.in1d(np.arange(1902, 2022), target_years + j)
        subset_years2 = np.in1d(target_years + j, np.arange(1902, 2022))

        target = target_values[subset_years2][np.newaxis]
        source = modelled_chronologies[:, subset_years]

        # Shorten chronologies around the centre if the set max_len is shorter than the sequence length
        start = int(len(target[0])/2) - int(max_len/2)
        stop = int(len(target[0])/2) + int(max_len/2)

        if start < 0:
            start = 0

        # Calculate R squared for all combinations of this offset
        r = 1 - cdist(target[:, start:stop],
                      source[:, start:stop], metric='correlation')[0]

        # Store results of this offset
        pearson_r.extend(r.tolist())
        offset.extend(np.repeat(j, len(r)))
        lat_new.extend(lat)
        lon_new.extend(lon)

    # Create a data frame with the longitude, latitude, and R-squared values, and sort it by R-squared value
    df = pd.DataFrame({'lon': lon_new, 'lat': lat_new,
                       'pearson_r': pearson_r, 'offset': offset}).sort_values(by=['pearson_r'], ascending=False)

    # Set Pearson R < 0 to 0, excluding negative correlations, and calculate R squared
    df.loc[df.pearson_r < 0, 'pearson_r'] = 0
    df.pearson_r = df.pearson_r ** 2
    df.rename(columns={'pearson_r': 'rsq'}, inplace=True)

    # Check for duplicates in the top five. After this, the top five consists of results that are either different years
    # and/or over 250 kilometres apart from eachother
    for j in range(5):
        df['dist_to' + str(j)] = vectorized_haversine(df.lon, df.lat, np.repeat(df.loc[df.index[j], 'lon'], len(df)),
                                                      np.repeat(df.loc[df.index[j], 'lat'], len(df)))

        mask = (df.offset != df.offset.iloc[j]) | (df['dist_to' + str(j)] > 250)
        mask.iloc[j] = True
        df = df[mask]

    # Get the ground truth longitude and latitude of the current record
    lat_ground_truth = record['latitude']
    lon_ground_truth = record['longitude']

    # Calculate the distance between all results and the ground truth
    df['dist'] = vectorized_haversine(df.lon, df.lat,
                                      np.repeat(lon_ground_truth, len(df)), np.repeat(lat_ground_truth, len(df)))

    # Save the top 1000 results to file (note that only the top five has the 'different year and/or different location'
    # conditions. Saving the top 1000 to file is optional and mainly to be used for result interpretation and debugging.
    df.head(1000).to_csv('3_2_both_unknown_out/' + record['name'] + '.csv', sep=';', index=False)

    # Create a boolean series of indices in the top five that are a match
    top5_series = (df.loc[df.index[0:5], 'dist'] < 250) & (df.loc[df.index[0:5], 'offset'] == 0)

    # Check if the top 1 result is a match, if so print Top 1
    if (df.loc[df.index[0], 'dist'] < 250) & (df.loc[df.index[0], 'offset'] == 0):
        top_results.append('Top 1')
    # Check if there are valid results in the top 5 series, if so print Top 5
    elif top5_series[top5_series].size > 0:
        top_results.append('Top 5')
    # Else print Not in top 5
    else:
        top_results.append('Not in top 5')

    rsq_top1.append(df.loc[df.index[0], 'rsq'])

# Add the results to the overview table
overview['both_unknown_top_result'] = top_results
overview['both_unknown_rsq_top1'] = rsq_top1

# Write new overview table
overview.to_csv('overview_after_4_3_sub.csv', sep=';', index=False)
