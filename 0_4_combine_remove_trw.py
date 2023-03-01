import pandas as pd


# Removes rows from the overview table
def remove(df, names):
    # Iterate over the names in the 'names' list
    for name in names:
        # Remove the rows with the specified name from the dataframe
        df = df[df.name != name]

    # Return the modified dataframe
    return df


# Combines rows the overview table, and also combines raw trw files
def combine(df, names):
    # Iterate over the names in the 'names' list
    for name in names:
        # Try to read the file with the specified name and concatenate it to the 'base' dataframe
        try:
            new = pd.read_csv('1_1_prepared_trw/' + str(name) + '.csv', sep=';', index_col=0, header=None)
            base = pd.concat([base, new], axis=1)
        # If the 'base' dataframe has not been defined yet, read the file and assign it to the 'base' dataframe
        except NameError:
            base = pd.read_csv('1_1_prepared_trw/' + str(name) + '.csv', sep=';', index_col=0, header=None)

    # Write the resulting 'base' dataframe to a new combined raw trw file with the combined names as the filename
    base.to_csv('1_1_prepared_trw/' + '+'.join(names) + '.csv', sep=';', header=False, index=True)

    # Create a new row for the overview table
    new_row = pd.DataFrame({'name': '+'.join(names),
                            'latitude': df[df.name == names[0]].latitude,
                            'longitude': df[df.name == names[0]].longitude,
                            'first_year': df[df.name.isin(names)].first_year.min(),
                            'last_year': df[df.name.isin(names)].last_year.max()})

    # Concatenate the new row to the overview table
    df = pd.concat([df, new_row])

    # Remove the rows with the old names from the overview table
    df = remove(df, names)

    # Return the modified overview table
    return df


# Read the the overview table
overview = pd.read_csv('overview_after_0_3.csv', sep=';')

# Contain only early- and/or latewood measurements >> remove
overview = remove(overview, ['germ012l', 'lith011e', 'lith011l'])

# Not located on land >> remove
overview = remove(overview, ['brit10'])

# Temporally too short >> remove
overview = remove(overview, ['fran048'])

# Stated in correlation stats that they are not useful; too many problems/flags/misdated samples >> remove
overview = remove(overview, ['neth022', 'neth023', 'neth024', 'neth028', 'neth029', 'neth030'])

# Are geographically very close, and are combined into a single chronology here (no duplicate samples) >> combine
overview = combine(overview, ['germ168', 'germ169'])
overview = combine(overview, ['germ195', 'germ196'])
overview = combine(overview, ['fran005', 'fran007'])

# Re-sort the overview table
overview = overview.sort_values(by=['name'])

# Write new updated overview table
overview.to_csv('overview_after_0_4.csv', sep=';', index=False)
