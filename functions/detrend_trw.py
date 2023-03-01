import os
import warnings

import numpy as np
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from astropy.utils.exceptions import AstropyWarning

# Catches AstropyWarnings, which in this case are caused by non-problematic code
warnings.simplefilter('ignore', category=AstropyWarning)


# Detrends TRW files given their names
def detrend(names, outdir, sigma=4, overwrite=True):
    # Loop over the supplied names and construct their path and the output path
    for name in names:
        in_path = os.path.join('1_1_prepared_trw', name) + '.csv'
        out_path = os.path.join(outdir, name) + '_sigma_' + str(sigma) + '.npy'

        # Check if the output file exists, and if it does not need to be overwritten, continue to the next file
        if (not overwrite) and (os.path.exists(out_path)):
            continue

        # Open the source file (prepared trw, so mostly raw data)
        source = np.genfromtxt(in_path, delimiter=';')

        # Remove values lower than or equal to 0 and remaining stop markers (999 or 9990)
        source[source <= 0] = np.nan
        source[source == 999] = np.nan
        source[source == 9990] = np.nan

        # Calculate trend column-wise
        trend = source.copy()
        for i, col in enumerate(trend.T):
            trend[:, i] = convolve(col, Gaussian1DKernel(stddev=sigma), boundary='extend', preserve_nan=True)

        # Remove values equal to zero in lower than or equal to zero in the trend table
        trend[trend <= 0] = np.nan

        # Detrend the source table by dividing by the trend table and log-transform
        out = np.log10(source / trend)

        # Reset the years column
        out[:, 0] = source[:, 0]

        # Remove the first and last two non-nan values from each column
        for i in range(2):
            out[np.isnan(out).argmin(axis=0), np.arange(out.shape[1])] = np.nan
            out = np.flip(out)
            out[np.isnan(out).argmin(axis=0), np.arange(out.shape[1])] = np.nan
            out = np.flip(out)

        # Remove all empty columns and rows
        out = out[~np.all(np.isnan(out[:, 1:]), axis=1), :]
        out = out[:, ~np.all(np.isnan(out), axis=0)]

        # Remove outliers column-wise (lower or higher than 4 stds)
        for column in out.T[1:]:
            std = np.nanstd(column)
            mean = np.nanmean(column)

            column[((mean - 4 * std) > column) | ((mean + 4 * std) < column)] = np.nan

        # Write to file
        np.save(out_path, out)
