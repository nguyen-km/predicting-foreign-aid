import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
import seaborn as sns

#Load data
df = pd.read_csv('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data/final_clean_data.csv', 
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
plt.title("Elbow Method")
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

# K-means PCA (for plotting)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame(data = X_pca
             , columns = ['principal component 1', 'principal component 2'])
df_pca['cluster'] = y_pred + 1

df_pca.head()

sns.scatterplot(data=df_pca, x='principal component 1', y = 'principal component 2', hue = 'cluster', 
                palette=['blue', 'red', 'green'])
plt.title("Cluster plot")
plt.show()
