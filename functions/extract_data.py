import numpy as np
import rasterio
from netCDF4 import Dataset
import warnings


# Extracts interpolated CRU TS meteo data given a list of tuples of coordinates
def get_interp_cru_ts(coords):
    print("Extracting monthly meteo data from CRU_TS...")
    # Initiate output table
    out = np.zeros([len(coords), 121, 4 * 12])

    # Loop over the variables of interest
    for i, variable in enumerate(['tmn', 'tmx', 'tmp', 'pre']):
        # Construct variable filename and open the dataset
        file = f'./0_2_CRU_TS/cru_ts4.06.1901.2021.{variable}.dat.nc'
        data = Dataset(file)
        var_data = data.variables[variable][:]

        # Loop over the coordinates
        for j, coord in enumerate(coords):
            x_coord = coord[0]
            y_coord = coord[1]

            # Introduce slight offset for literal edge cases. This is a code-efficient workaround, with next to no
            # impact on the results
            if (x_coord + 0.25) % 0.5 == 0:
                x_coord += 0.000001

            if (y_coord + 0.25) % 0.5 == 0:
                y_coord += 0.000001

            # Get the nearest two x indices and distances to the coord of interest
            x_distances = np.abs(data.variables['lon'][:] - x_coord).data
            nearest_x_indices = np.argpartition(x_distances, 2)[:2]
            nearest_x_distances = x_distances[nearest_x_indices]

            # Get the nearest two y indices and distances to the coord of interest
            y_distances = np.abs(data.variables['lat'][:] - y_coord).data
            nearest_y_indices = np.argpartition(y_distances, 2)[:2]
            nearest_y_distances = y_distances[nearest_y_indices]

            # Create arrays of the values of the nearest x- and y-pixels to the coord of interest
            nearest_x_pixels, nearest_y_pixels = np.meshgrid(nearest_x_indices, nearest_y_indices)
            nearest_x_pixels = nearest_x_pixels.flatten()
            nearest_y_pixels = nearest_y_pixels.flatten()

            # Create arrays of the distances of the nearest x- and y-pixels to the coord of interest
            mesh_x_distances, mesh_y_distances = np.meshgrid(nearest_x_distances, nearest_y_distances)
            mesh_x_distances = mesh_x_distances.flatten()
            mesh_y_distances = mesh_y_distances.flatten()

            # Calculate the distances (diagonal, using Pythagoras theorem)
            distances = np.sqrt(mesh_x_distances ** 2 + mesh_y_distances ** 2)[:, None]

            # Initiate empty array for the data values
            values = np.empty((4, var_data.shape[0]), dtype=var_data.dtype)

            # Calculate the values corresponding to these distances and put into empty array
            for k, (x, y) in enumerate(zip(nearest_x_pixels, nearest_y_pixels)):
                values[k] = var_data[:, y, x]

            # Keep only valid values (very high values indicate missing data here)
            valid_indices = np.where(values[:, 0] < 999999)
            values = values[valid_indices]
            distances = distances[valid_indices]

            # Apply inverse distance weighting with p=2
            p = 2
            weights = np.divide(1, distances**p)
            weighted_mean = np.divide(np.sum(values * weights, axis=0), np.sum(weights, axis=0))

            # Add result to output table
            out[j, :, i * 12: (i + 1) * 12] = np.reshape(weighted_mean, (121, 12))

    return out


# Extracts worldclim meteo data given a list of tuples of coordinates
def get_worldclim(coords):
    print("Extracting long term high resolution meteo data from Wordclim...")
    # Initiate output table
    out = np.zeros([len(coords), 4 * 12])

    # Loop over variables of interest
    for i, var in enumerate(['tmin', 'tmax', 'tavg', 'prec']):
        # Loop over all available months
        for j in range(12):
            # Open file, sample the given coords and add them to the output table
            month = '{:02d}'.format(j + 1)
            src = rasterio.open('./0_3_worldclim/wc2.1_30s_' + var + '/wc2.1_30s_' + var + '_' + month + '.tif')
            out[:, i * 12 + j] = [x[0] for x in src.sample(coords)]

    # Warn user if any of the values contain nodata (indicating the coords are outside worldclim coverage)
    if np.any(out < -1e37):
        warnings.warn('At least one of the coordinates is outside WorldClim boundaries!')

    return out


# Returns downscaled CRU TS data using a high spatial resolution long term worldclim correction
def get_downscaled_cru_ts(coords):
    # Get the CRU TS and worldclim data
    cru_ts = get_interp_cru_ts(coords)
    worldclim = get_worldclim(coords)

    print('Downscaling CRU_TS using Worldclim...')
    # Loop over each coordinate in the CRU TS data
    for x, point in enumerate(cru_ts):
        # Loop over each variable
        for i, var in enumerate(['tmin', 'tmax', 'tavg', 'prec']):
            # Loop over each month
            for j in range(12):
                # Get the CRU TS average for the given combination of coordinate, variable and month over the same
                # temporal coverage as worldclim
                col = i * 12 + j
                avg_point = np.mean(point[70:101, col])

                # For temperature variables, use subtraction of worldclim data as a means of correction
                if var == 'tmin' or var == 'tmax' or var == 'tavg':
                    diff = avg_point - worldclim[x, col]
                    point[:, col] = point[:, col] - diff
                # For precipitation, use division by worldclim data as a means of correction
                elif var == 'prec':
                    # Only perform the correction if the worldclim precipitation is over 0
                    if worldclim[x, col] > 0:
                        diff = avg_point / worldclim[x, col]
                        point[:, col] = point[:, col] / diff

    return cru_ts


# Extracts soil data from soilgrids rasters given a list of tuples of coordinates
def get_soil(coords):
    print("Extracting soil data from Soilgrids...")

    # Initiate output table
    out = np.zeros([len(coords), 5])

    # Loop over the variables of interest
    for i, var in enumerate(['clay', 'sand', 'silt', 'soc', 'nitrogen']):
        # Open the corresponding file
        src = rasterio.open('./0_4_soilgrids/prepared/' + var + '_15_30cm_prepared.tif')

        # Extract the values for each coordinate and write to output table
        out[:, i] = [x[0] for x in src.sample(coords)]

    return out
