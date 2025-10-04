import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Spotify_Dataset_V3.csv', delimiter=';')

sample_features = df[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence']]
scaler = StandardScaler()
sample_scaled_features = scaler.fit_transform(sample_features)

pca = PCA(n_components=2)
sample_pca_transformed_features = pca.fit_transform(sample_scaled_features)

gmm = GaussianMixture(n_components=6, random_state=42)
gmm_clusters = gmm.fit_predict(sample_pca_transformed_features)

df['Cluster'] = gmm_clusters

df.to_csv('Spotify_With_Clusters.csv', index=False, sep='^')
print("Exported dataframe with clusters to 'Spotify_With_Clusters.csv'.")

plt.figure(figsize=(10, 6))
for cluster in set(gmm_clusters):
    plt.scatter(
        sample_pca_transformed_features[gmm_clusters == cluster, 0],
        sample_pca_transformed_features[gmm_clusters == cluster, 1],
        label=f'Cluster {cluster}', alpha=0.6
    )
plt.title('Gaussian Mixture Model Clustering on PCA-Reduced Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title="Clusters")
plt.show()