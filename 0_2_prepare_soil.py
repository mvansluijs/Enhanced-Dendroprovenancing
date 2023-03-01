import gdal
import rasterio
from rasterio.fill import fillnodata


# Fills in missing soil data using interpolation and masks data to match Worldclim coverage
def fill_and_mask(var):
    # Construct input and output filenames based on the given variable name
    infile = '0_4_soilgrids/source/' + var + '_15_30cm.tif'
    outfile = '0_4_soilgrids/prepared/' + var + '_15_30cm_prepared.tif'

    # Worldclim mask filename (any of them can be chosen here, as they all have the same coverage + extent)
    maskfile = '0_3_worldclim/wc2.1_30s_prec/wc2.1_30s_prec_01.tif'

    # Use rasterio to fill in missing data in the input file using IDW interpolation
    with rasterio.open(infile) as src:
        # Get the profile (metadata) from the source file
        profile = src.profile
        # Read the data from the source file
        arr = src.read(1)
        # Fill in missing data using the mask from the source file
        arr_filled = fillnodata(arr, mask=src.read_masks(1), max_search_distance=10, smoothing_iterations=0)

    # Write the filled data to the output file
    with rasterio.open(outfile, 'w', **profile) as dest:
        dest.write_band(1, arr_filled)

    # Use GDAL to mask the data in the output file using the mask file
    # Open the output file in edit mode
    ds = gdal.Open(outfile, 1)
    # Get the band (data layer) from the output file
    band = ds.GetRasterBand(1)
    # Read the data from the output file
    data = ds.ReadAsArray()
    # Open the mask file
    ds2 = gdal.Open(maskfile)
    # Read the data from the mask file
    mask = ds2.ReadAsArray()
    # Get the nodata value from the output file
    ndval = band.GetNoDataValue()

    # Make sure the mask and source file have the same shape
    assert data.shape == mask.shape

    # Set the soil data to nodata if worldclim contains nodata at that pixel
    data[mask == -32768] = ndval

    # Write and close the datasets
    band.WriteArray(data)
    ds.FlushCache()
    band = None
    ds = None
    del data
    del mask
    ds2 = None


# Iterate over all soil variables and apply the fill_and_mask function to each
for var_name in ['sand', 'clay', 'silt', 'soc', 'nitrogen']:
    fill_and_mask(var_name)
