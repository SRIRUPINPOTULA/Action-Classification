from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def kMeans(data, latent_count):
    # Apply KMeans to get the cluster centers
    kmeans = KMeans(n_clusters=latent_count, random_state=42)
    kmeans.fit(data)

    # Get the Cluster Centers
    cluster_centers = kmeans.cluster_centers_

    latent_model = cdist(data, cluster_centers, 'euclidean')

    return latent_model
