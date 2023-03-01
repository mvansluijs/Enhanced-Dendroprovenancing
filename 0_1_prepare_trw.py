import os
import pandas as pd


# Converts a -noaa.rwl file to a more usable, tabular, format
def convert_files(name, path):
    # Read the file as a Pandas data frame, ignoring lines that start with '#' (the header)
    df = pd.read_table(path, comment='#')

    # Convert all values in the dataframe to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop rows where the 'age_CE' column is less than 1850
    df.drop(df[df.age_CE < 1850].index, inplace=True)

    # Drop rows that contain less than 2 trw values (years with less than 2 measurements left)
    df.dropna(thresh=2, inplace=True)

    # Drop columns that have less than 15 trw values (trw samples shorter than 15 years)
    df.dropna(axis=1, thresh=15, inplace=True)

    # Write the converted table to file
    df.to_csv('1_1_prepared_trw/' + name[:-9] + '.csv', sep=';', header=False, index=False)


# Scan the raw trw file directory for files that end with '-noaa.rwl' and convert
with os.scandir('0_1b_trw_raw') as it:
    for entry in it:
        if entry.name.endswith('-noaa.rwl') and entry.is_file():
            convert_files(entry.name, entry.path)
