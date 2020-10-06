import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("agaricus-lepiota.csv")[0:5000]

X = pd.get_dummies(df)
X = pd.DataFrame(preprocessing.scale(X), columns=X.columns)

start = 2

runs = 30

skip = 1

points = np.empty((runs - start),)

for k in range(start, runs):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(X)
    points[k - start] = np.array([silhouette_score(X, kmeans.labels_, metric='euclidean')])

# Visualise the different accuracies at k=2...30
m = np.amax(points, axis=0)
k = points.argmax() + start
print("Max: %.4f at k=%i" % (m, k))
plt.figure()
plt.plot(range(start, runs), points)
label = '(%i,%.4f)' % (k, m)
plt.plot([k], [m], c='red', marker='o', markersize=10, label=label)
plt.xlabel('Number of clusters k')
plt.ylabel('Accuracy score')
plt.legend()

# use PCA to select features
pca = PCA(n_components=k)
X_PCA = pca.fit_transform(X)

# Visualise the data using k=best from k=2...30
kmeans = KMeans(n_clusters=k)
kmeans.fit(X_PCA)
y_kmeans = kmeans.predict(X_PCA)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X_PCA[:, 0], X_PCA[:, 1], zs=X_PCA[:, 2], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], zs=centers[:, 2], c='black', s=500, alpha=0.75)

plt.show()


