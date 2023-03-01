import numpy as np
import fiona
from shapely.geometry import shape, Point
from shapely.prepared import prep
import rasterio

# Initiate valid points list
valid_points = []

# Open distribution shapefile
shp = fiona.open('0_5_species_distribution/Quercus_robur_plg_clip.shp')

# Loop over polygons in the distribution shapefile
for polygon in shp:
    polygon_geom = shape(polygon['geometry'])

    # Get geometry boundaries
    latmin, lonmin, latmax, lonmax = polygon_geom.bounds

    # Set desired resolution
    resolution = .2

    # Construct a rectangular mesh based on boundaries and resolution
    points = []
    for lat in np.arange(int(latmin), int(latmax+1), resolution):
        for lon in np.arange(int(lonmin), int(lonmax+1), resolution):
            points.append(Point((lat, lon)))

    # Create prepared polygon
    prep_polygon = prep(polygon_geom)

    # Validate if each point of the rectangular mesh falls inside shape using the prepared polygon and if so, extend
    # the valid points list with those points
    valid_points.extend(filter(prep_polygon.contains, points))

# Convert valid_points to a numpy array by putting the valid points into a list of lists first
list_of_lists = []
for pp in valid_points:
    list_of_lists.append([pp.x, pp.y])
out = np.unique(np.array(list_of_lists), axis=0)

# Find the coordinates that are outside of the WorldClim mask (and thus also outside the soil data)
src = rasterio.open('./0_3_worldclim/wc2.1_30s_tmin/wc2.1_30s_tmin_01.tif')
coords = [(x, y) for x, y in zip(out[:, 0], out[:, 1])]
mask = np.array([x[0] for x in src.sample(coords)]) > -1e37

# Filter out these coordinates outside WorldClim coverage
out = out[mask]

# Save array to file
np.save('prediction_points.npy', out)
