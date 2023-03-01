import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load training table (here for sigma 4.0)
training_table = np.load('training_table_sigma_4.0.npy')

# Select values to train on (all but current chronology) and split into inputs and targets (crn values, 1st column)
inputs = np.delete(training_table, 0, 1)
targets = training_table[:, 0]

# Fit random forest
rf = RandomForestRegressor(n_jobs=-1, n_estimators=500, random_state=42)
rf.fit(inputs, targets)

# Create empty numpy array for storing the results
out = np.empty([np.size(training_table, 1) - 1, 6])
out.fill(np.nan)

# Transform to Pandas dataframe
out = pd.DataFrame(out, columns=['group', 'var', 'type', 'offset', 'month', 'importance'])

# Filling the first column (group)
out.iloc[:288, 0] = 'meteo'
out.iloc[288:293, 0] = 'soil'
out.iloc[293, 0] = 'year'
out.iloc[294:296, 0] = 'coords'
out.iloc[296:, 0] = 'one_hot'

# Filling the second column (var)
out.iloc[:288, 1] = np.tile(np.repeat(np.array(['tmin', 'tmax', 'tavg', 'prec'], dtype=str), 12), 6)
out.iloc[288, 1] = 'clay'
out.iloc[289, 1] = 'sand'
out.iloc[290, 1] = 'silt'
out.iloc[291, 1] = 'soc'
out.iloc[292, 1] = 'nitrogen'
out.iloc[294, 1] = 'lat'
out.iloc[295, 1] = 'lon'

# Filling the rest of the columns (type, offset, month, importance)
out.iloc[:288, 2] = np.tile(np.repeat(np.array(['value', 'moving_avg', 'deviation'], dtype=str), 48), 2)
out.iloc[:288, 3] = np.repeat(np.array([0, -1]), 144)
out.iloc[:288, 4] = np.tile(np.arange(1, 13), 24)
out.iloc[:, 5] = rf.feature_importances_

out.to_csv('4_0_general_outputs/feature_importances.csv', sep=';', index=False)
