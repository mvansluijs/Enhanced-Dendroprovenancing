import matplotlib.pyplot as plt
import pandas as pd

# Load overview table
overview = pd.read_csv('overview_after_4_3.csv', sep=';')

# Get data
top5 = overview['both_unknown_rsq_top1'][overview['both_unknown_top_result'] != "Not in top 5"]
not_top5 = overview['both_unknown_rsq_top1'][overview['both_unknown_top_result'] == "Not in top 5"]

# Plot
plt.boxplot([top5, not_top5], widths=.6, labels=['In top 5', 'Not in top 5'], patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5), boxprops=dict(facecolor='#3F88FF'))
plt.ylabel("Top 1 R$^2$ value", fontsize=16)
plt.xticks(fontsize=12)
plt.show()
