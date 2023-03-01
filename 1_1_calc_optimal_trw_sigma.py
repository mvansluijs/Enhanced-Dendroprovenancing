import numpy as np
import pandas as pd
import statsmodels.api as sm

from functions.detrend_trw import detrend
from functions.extract_data import get_downscaled_cru_ts

# Load the overview table
overview = pd.read_csv('overview_after_0_4.csv', sep=';')

# Extract the longitude and latitude columns from the overview table
longitude = overview.longitude
latitude = overview.latitude

# Create a list of tuples containing the longitude and latitude values
train_coords = [(x, y) for x, y in zip(longitude, latitude)]

# Get meteo data
meteo = get_downscaled_cru_ts(train_coords)

# Initiate overall results list
results = []

# Loop through a range of possible sigmas from 0.2 to 10 with increments of 0.2
for sigma in np.linspace(.2, 10, 50):
    # Workaround for numpy float problems (relevant later in filenaming)
    sigma = round(sigma, 1)

    # Pass all files with names in the overview table to be detrended (and saved) using the current sigma value
    detrend(overview['name'], '1_2_detrended_trw', sigma=sigma, overwrite=False)

    # Initiate results list for the current sigma value
    sigma_rsquared = []

    # Loop through the names (/sites) in the overview table
    for i in overview.index:
        code = overview.at[i, 'name']
        meteo_point = meteo[i]

        # Get the average temperatures for the current site
        tavg = pd.DataFrame(np.mean(meteo_point[:, 24:36], axis=1))
        tavg.index = range(1901, 2022)

        # Get the precipitation for the current site
        prec = pd.DataFrame(np.mean(meteo_point[:, 36:48], axis=1))
        prec.index = range(1901, 2022)

        # Read the detrended trw file for the current site
        trw = pd.DataFrame(np.load('1_2_detrended_trw/' + str(code) + '_sigma_' + str(sigma) + '.npy'))

        # Initiate results list for the current site
        rsquared_site = []

        # Go through each sample and perform an OLS regression between the sample and the precipitation and temperature
        for j in range(len(trw.columns) - 1):
            trw_sub = trw.iloc[:, [0, j + 1]]
            trw_sub = trw_sub.set_index(list(trw_sub)[0])

            combined = pd.merge(trw_sub, tavg, left_index=True, right_index=True).dropna()
            combined = pd.merge(combined, prec, left_index=True, right_index=True).dropna()

            # Only consider samples that have over three measurements
            if len(combined) > 3:
                X = sm.add_constant(combined.iloc[:, 1:])
                model = sm.OLS(combined.iloc[:, 0], X).fit()

                # Save R squared
                rsquared_site.append(model.rsquared)

        # Calculate the median R squared for the current site and add to the results list of the current sigma
        sigma_rsquared.append(np.nanmedian(rsquared_site))

    # Add the R squared list of the current sigma to the results list
    results.append(sigma_rsquared)
    print('sigma=' + str(sigma) + ' done')

# Convert results list to a Pandas table, and adjust index and column names
results_df = pd.DataFrame(results)
results_df.index = np.linspace(.2, 10, 50)
results_df.columns = overview.name
df = results_df.T

# Initiate a list for optimal sigmas
optimal_sigmas = []

# Loop over all names (/sites) and calculate the sigma to be used, based on the best performing sigma in all other sites
for name in overview.name:
    part_df = df[~np.array(df.index == name)]
    optimal = part_df.mean(axis=0).rolling(5, min_periods=1, center=True).mean().idxmax()
    optimal_sigmas.append(optimal)

# Add the sigma value to be used for each site to the overview table
overview['use_sigma'] = optimal_sigmas
overview.to_csv('overview_after_1_1.csv', sep=';', index=False)
