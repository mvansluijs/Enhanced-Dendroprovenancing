import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Import data
X = [10, 20, 30, 40, 50, 'all']
top1 = [0.121495327, 0.392523364, 0.523364486, 0.644859813, 0.710280374, 0.822429907]
top5 = [0.252336449, 0.61682243, 0.747663551, 0.803738318, 0.85046729, 0.869158879]

# Specify x-axis array
X_axis = np.arange(len(X))

# Plot bars
plt.bar(X_axis - 0.2, top1, 0.4, label='Top 1', facecolor='#9EC1A3')
plt.bar(X_axis + 0.2, top5, 0.4, label='Top 5', facecolor='#3F88FF')

# Set y-axis to percentage
ax = plt.gca()
ax.set_ylim([0, 1])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# Plot
plt.xticks(X_axis, X)
plt.xlabel("Years used for prediction", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.legend()
plt.show()
