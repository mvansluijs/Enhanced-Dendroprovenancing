import matplotlib.pyplot as plt
import pandas as pd

# Load overview table
overview = pd.read_csv('overview_after_4_3.csv', sep=';')

# Select location unknown distance column
data = overview['location_unknown_distance']

# Set binwidth
binwidth = 100

# Plot histogram
plt.hist(data, bins=range(0, int(max(data) - (max(data) % binwidth) + 2 * binwidth), binwidth), facecolor='#3F88FF')
plt.xlabel('Distance between prediction & ground truth (in km)', fontsize=16)
plt.ylabel('Chronologies (count)', fontsize=16)
plt.grid(True)
plt.show()
