import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm

sp500_data = pd.read_csv("../csv/monthly_returns_snp500.csv", index_col=0)
markowitz_data = pd.read_csv("../csv/monthly_returns_markowitz.csv", index_col=0)
kmeans_data = pd.read_csv("../csv/monthly_returns_kmeans.csv", index_col=0)
kmeans_dtw_data = pd.read_csv("../csv/monthly_returns_kmeans_dtw.csv", index_col=0)
kshape_data = pd.read_csv("../csv/monthly_returns_kshape.csv", index_col=0)

# IQR 이상치 제거
Q1 = markowitz_data.quantile(0.25)
Q3 = markowitz_data.quantile(0.75)
IQR = Q3 - Q1

# 이상치 조건
outlier_condition = (markowitz_data < (Q1 - 1.5 * IQR)) | (markowitz_data > (Q3 + 1.5 * IQR))

# 이상치를 NaN으로 대체
markowitz_data_filtered = markowitz_data.mask(outlier_condition)

combined_data_filtered = pd.DataFrame({
    'KShape': kshape_data.iloc[:, 0],
    'KMeans(DTW)': kmeans_dtw_data.iloc[:, 0],
    'KMeans': kmeans_data.iloc[:, 0],
    'Markowitz': markowitz_data_filtered.iloc[:, 0],
    'S&P500': sp500_data.iloc[:, 0],

})

# NaN 값 제거
combined_data_filtered.dropna(inplace=True)

names = combined_data_filtered.columns.tolist()

boxprops = dict(linestyle='-', linewidth=2, color='blue')
medianprops = dict(linestyle='-', linewidth=2, color='red')

# Boxplot
plt.figure(figsize=(12, 8))
bp = plt.boxplot(combined_data_filtered.values, vert=False, patch_artist=True,
                 boxprops=boxprops, medianprops=medianprops, labels=combined_data_filtered.columns)

# Making the interiors of the boxes transparent
for box in bp['boxes']:
    box.set(facecolor='none')

# Adding the data points
for i, name in enumerate(names):
    val = combined_data_filtered[name].tolist()
    xs = np.random.normal(i + 1, 0.04, len(val))
    plt.scatter(val, xs, c=cm.prism((i + 1) / len(names)), alpha=0.4)

plt.xlabel('Log Monthly Returns', fontsize=22)
plt.yticks(fontsize=22)
plt.show()