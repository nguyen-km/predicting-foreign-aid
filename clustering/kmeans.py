import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples

#Load data
path = 'final_clean_data.csv'
df = pd.read_csv(path, 
                index_col=0)
df.head()

X = df.select_dtypes(include=np.number) # Keep only numeric data
X
X.shape

# Elbow Method
inertia = []
n = 10
for k in range(1, n+1):
    kmeans = KMeans(k, n_init = 'auto')
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,n+1), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

k_opt = 3 # select optimal k at the 'elbow'
kmeans = KMeans(k_opt, n_init = 'auto')
kmeans.fit(X)

kmeans.cluster_centers_

# sihouette plot
y_pred = kmeans.fit_predict(X)
cluster_labels = np.unique(y_pred)

silhouette_vals = silhouette_samples(X, y_pred)

y_ax_lower, y_ax_upper = 0, 0
yticks = []
for label in cluster_labels:
    c_silhouette_vals = silhouette_vals[y_pred == label]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red",linestyle="--") # plot mean silhouette value as vertical line
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()
