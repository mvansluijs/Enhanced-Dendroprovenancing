import os
import warnings
import pandas as pd


# Extracts general site information from a -noaa.rwl files
def get_info(path):
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                # Extract the relevant information
                if 'Northernmost_Latitude' in line:
                    first, remainder = line.split('Northernmost_Latitude: ')
                    latitude = remainder[:-1]
                if 'Easternmost_Longitude:' in line:
                    first, remainder = line.split('Easternmost_Longitude: ')
                    longitude = remainder[:-1]
                if 'Earliest_Year' in line:
                    first, remainder = line.split('Earliest_Year: ')
                    first_year = remainder[:-1]
                if 'Most_Recent_Year:' in line:
                    first, remainder = line.split('Most_Recent_Year: ')
                    last_year = remainder[:-1]

    # Return the extracted information
    return latitude, longitude, first_year, last_year


# Open the 'overview_after_0_3.csv' file in write mode
with open('overview_after_0_3.csv', 'w+') as f:
    # Write a header line to the file
    f.write('name;latitude;longitude;first_year;last_year\n')

    # Scan the '0_1b_trw_raw' directory for files that end with '-noaa.rwl'
    with os.scandir('0_1b_trw_raw') as it:
        # Iterate over the entries in the directory
        for entry in it:
            # If the entry is a file and its name ends with '-noaa.rwl'
            if entry.name.endswith('-noaa.rwl') and entry.is_file():
                # Get the information from the file
                lat, lon, first_y, last_y = get_info(entry.path)

                # Write the information to the 'overview_after_0_3.csv' file
                f.write(entry.name[:-9] + ';' + lat + ';' + lon + ';' + first_y + ';' + last_y + '\n')

# Read the newly created overview table as a Pandas dataframe
overview = pd.read_csv('overview_after_0_3.csv', sep=';')

# Find duplicate rows in the overview table based on longitude and latitude
duplicates = overview[overview.duplicated(subset=['longitude', 'latitude'], keep=False)]

# If there are any duplicate rows in the duplicates dataframe display a warning message and print the duplicates
if len(duplicates) > 0:
    warnings.warn(
        'Some of the trw files have the same coordinates! See the table(s) below. Combine or remove these in 0_4_combine_remove_trw.',
        stacklevel=2)

    # Iterate over the groups in the 'duplicates' dataframe and print each group separately
    for name, group in duplicates.groupby(['longitude', 'latitude']):
        print(group, '\n')
