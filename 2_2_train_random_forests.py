import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# Load the overview table
overview = pd.read_csv('overview_after_1_2.csv', sep=';')

# Loop over all chronologies in the overview table (reversed due to the order of the one-hot encoding; back-to-front)
for i, code in tqdm(enumerate(reversed(overview['name'])), desc="Training random forests", smoothing=0, total=len(overview)):
    # Load the training table corresponding to the sigma that should be used
    training_table = np.load('training_table_sigma_' + str(overview.iloc[i]['use_sigma']) + '.npy')

    # Select values to train on (all but current chronology) and split into inputs and targets (crn values, 1st column)
    train = training_table[(training_table[:, -(i + 1)] != 1)]
    train_inputs = np.delete(train, 0, 1)
    train_targets = train[:, 0]

    # Fit random forest
    rf = RandomForestRegressor(n_jobs=-1, n_estimators=500, random_state=42)
    rf.fit(train_inputs, train_targets)

    # Save model to file
    joblib.dump(rf, '2_1_random_forests/rf_without_' + code + '.joblib', compress=3)
