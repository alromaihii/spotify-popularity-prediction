from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = 'data/spotify_data_with_features.csv' 
spotify_data = pd.read_csv(file_path)

features_for_clustering = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 
                           'Acousticness', 'Instrumentalness', 'Valence']

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(spotify_data[features_for_clustering])


train_data, test_data = train_test_split(scaled_features, test_size=0.2, random_state=42)


kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(train_data)


test_clusters = kmeans.predict(test_data)

silhouette_avg = silhouette_score(test_data, test_clusters)
print("Silhouette Score for Testing Set (20% of the dataset):", silhouette_avg)

pca = PCA(n_components=2)
train_pca = pca.fit_transform(train_data)
test_pca = pca.transform(test_data)

# Plot the training data clusters
plt.figure(figsize=(10, 6))
for cluster in range(kmeans.n_clusters):
    cluster_points = train_pca[kmeans.labels_ == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster + 1}', alpha=0.6)

# Mark cluster centroids
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

# Adding plot details
plt.title('K-Means Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()
