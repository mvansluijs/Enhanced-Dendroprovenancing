import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Catch warnings caused by using np.nanmean on empty slices later in the code. Using np.nanmean this way is actually
# non-problematic in practice, but the recurring warning is annoying in the console output.
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# Load the overview table and initiate a new version by creating a copy
overview = pd.read_csv('overview_after_1_1.csv', sep=';')

# Loop over the different options for sigma
for sigma in overview['use_sigma'].unique():
    # Initiate list for storing usable crn lengths
    crn_lengths = []

    new_overview = overview.copy()

    # Loop over all records in the overview table
    for i in range(len(overview)):
        # Get the current record and extract the name
        code = overview.iloc[i]['name']

        # Construct file path to be used and load the corresponding trw file
        in_path = '1_2_detrended_trw/' + str(code) + '_sigma_' + str(sigma) + '.npy'
        trw = np.load(in_path)

        # Loop over all non-year columns
        for x in range(len(trw.T[1:])):
            # If we have to iterate more than half of the amount of columns, over half of the TRW sequences has been
            # removed, and the trw file is considered internally bad and has been removed from further analysis
            if x > .5 * len(trw.T[1:]):
                warnings.warn(code + " + sigma " + str(sigma) + " is internally bad and has been removed from further "
                                                                "analysis", stacklevel=2)

                # Remove row from new overview table
                new_overview = new_overview.drop(i)

                # Break out of loop and continue with next TRW file
                break

            # Initiate the p_value for the worst correlating TRW sequence to the mean of the rest
            worst = 0

            # Loop over all columns except the year column
            for j in range(len(trw.T[1:])):
                # Extract column of interest and create a mean chronology using the rest
                test = trw.T[1 + j][np.newaxis]
                rest = np.nanmean(np.delete(trw.T[1:], j, 0), axis=0)

                # Combine the two in an array and filter out any years with nan values
                comb = np.vstack((test, rest))
                comb = comb[:, ~np.isnan(comb).any(axis=0)]

                # Perform a simple linear regression
                model = sm.OLS(comb[1], sm.add_constant(comb[0])).fit()
                p = model.pvalues[1]

                # If the current iteration has a larger p_value than previous iterations, store that p_value and the
                # index of the current iteration
                if p > worst:
                    worst = p
                    worst_j = j

            # If the highest (/worst) p_value is lower than 0.01, remove the corresponding TRW sequence/column
            if worst > 0.01:
                trw = np.column_stack((trw[:, 0], np.delete(trw[:, 1:], worst_j, 1)))
            # If not take the mean of all columns and add the year column
            else:
                crn = np.nanmean(trw[:, 1:], axis=1)
                out = np.column_stack((trw[:, 0], crn))

                # Export the resulting chronology to file
                out_path = '1_3_chronologies/' + code + '_sigma_' + str(sigma) + '_crn.csv'
                np.savetxt(out_path, out, delimiter=';')

                years_bool = trw[:, 0] >= 1902
                crn_lengths.append(len(trw[years_bool]))

                # Break out of loop and continue with next TRW file
                break

    # Add usable crn lengths to new overview table
    new_overview['usable_crn_length'] = crn_lengths

    # Write new overview table to file
    new_overview.to_csv('overview_after_1_2.csv', sep=';', index=False)
