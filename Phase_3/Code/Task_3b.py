import os
import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class VideoSearchTool:
    def __init__(self, db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir, dimensionality_reduction, Feature_Space_Map):
        self.db_path = db_path
        self.feature_column = feature_column
        self.dimensionality_reduction = dimensionality_reduction
        self.num_layers = num_layers
        self.hashes_per_layer = hashes_per_layer # Hash functions per LSH layer
        self.w = w # Bucket width for LSH
        self.Feature_Space_Map = Feature_Space_Map  # Mapping of feature columns
        self.thumbnail_dir = thumbnail_dir # Directory containing video thumbnails
        self.lsh_index = {layer: {} for layer in range(num_layers)} #initialise LSH index
        self.video_features = {} # Dictionary to store video features
        self.load_data() #to get preprocessed data

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        feature_column = self.Feature_Space_Map[self.feature_column]
        query = f"SELECT videoID, {self.Feature_Space_Map[self.feature_column]} FROM data"
        #print("The query is: ", query)
        raw_features = []
        video_ids = []
        
        for row in cursor.execute(query):
            videoID, feature_data = row
            if feature_column in ["Layer_3", "AvgPool"]:
                feature_vector = np.array(json.loads(feature_data)) # Parse JSON feature
            else:
                feature_data = feature_data.strip("[]").split()
                feature_vector = np.array(feature_data, dtype=int) # Parse space-separated data
            raw_features.append(feature_vector)
            video_ids.append(videoID)
        
        conn.close()

        # Ensure feature vectors have consistent size
        processed_arrays = []
        for arr in raw_features:
            if len(arr) < 480:
                padded_arr = np.pad(arr, (0, 480 - len(arr)), mode='constant', constant_values=0)
            elif len(arr) > 480:
                padded_arr = arr[:480]
            else:
                padded_arr = arr
            
            processed_arrays.append(padded_arr)

        raw_features = np.array(processed_arrays)
        reduced_features = self.apply_dimensionality_reduction(raw_features, self.dimensionality_reduction)
        # Store processed features and add to LSH index
        for video_id, feature_vector in zip(video_ids, reduced_features):
            self.video_features[video_id] = feature_vector
            self.add_to_lsh(video_id, feature_vector)

    def apply_dimensionality_reduction(self, data, dimensionality_reduction):
        if dimensionality_reduction=='PCA':
            # Perform PCA to reduce dimensionality
            row, column = data.shape
            latent_count = min(256, column)
            cov_matrix = np.cov(data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors_subset = eigenvectors[:, sorted_indices[:latent_count]]
            return np.dot(data, eigenvectors_subset)
        elif dimensionality_reduction=='SVD':
            DtD = np.dot(data.T, data)

            # Calculate eigenvalues and eigenvectors for D^T D
            eigenvalues_V, V = np.linalg.eigh(DtD)
            latent_count = 256
            # Sort the eigenvalues to get the top latent semantics
            sorted_indices = np.argsort(eigenvalues_V)[::-1]
            eigenvalues_V = eigenvalues_V[sorted_indices]
            V = V[:, sorted_indices]
                
            V_subset = V[:, :latent_count]

            # Data in Reduced Dimensional space
            svd_data = np.dot(data, V_subset)

            return svd_data
        
        else:
            latent_count = 256
            kmeans = KMeans(n_clusters=latent_count, random_state=42)
            kmeans.fit(data)
            
            # Get the Cluster Centers
            cluster_centers = kmeans.cluster_centers_
            latent_model = cdist(data, cluster_centers, 'euclidean')
            return latent_model
    
    def add_to_lsh(self, video_id, feature_vector):
        for layer in range(self.num_layers):
            # Generate random hyperplanes and biases for hashing
            rp = np.random.randn(self.hashes_per_layer, feature_vector.shape[0])
            biases = np.random.uniform(0, self.w, size=self.hashes_per_layer)
            
            # Compute hash keys for the feature vector
            hash_key = tuple(
                int(np.floor((np.dot(feature_vector, hyperplane) + bias) / self.w))
                for hyperplane, bias in zip(rp, biases)
            )
            
            if hash_key not in self.lsh_index[layer]:
                self.lsh_index[layer][hash_key] = []
            self.lsh_index[layer][hash_key].append(video_id)
        #print(f"Layer {layer} LSH index size: {len(self.lsh_index[layer])}")
    
    def euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2) # Get euclidean distance

    def search(self, query_video_id, t):
        query_features = self.video_features[query_video_id]
        print("THe query features are: ", query_features)
        candidates = set() # Unique candidates from LSH index
        overall_candidates = 0
        
        for layer in range(self.num_layers):
            rp = np.random.randn(self.hashes_per_layer, query_features.shape[0])
            biases = np.random.uniform(0, self.w, size=self.hashes_per_layer)
            
            hash_key = tuple(
                int(np.floor((np.dot(query_features, hyperplane) + bias) / self.w))
                for hyperplane, bias in zip(rp, biases)
            )
            
            if hash_key in self.lsh_index[layer]:
                candidates.update(self.lsh_index[layer][hash_key])
                overall_candidates += len(self.lsh_index[layer][hash_key])
        # Compute similarity scores for candidates
        similarity_scores = [
            (video_id, self.euclidean_distance(query_features, self.video_features[video_id]))
            for video_id in candidates
        ]

        similarity_scores.sort(key=lambda x: x[1]) # Sort by distance
        
        top_videos = similarity_scores[:t] # Retrieve top t videos

        print(f"Unique Candidates: {len(candidates)}")
        print(f"Overall Candidates: {overall_candidates}")
        for video_id, score in top_videos:
            print(f"Video {video_id}: Distance {score:.4f}")
        
        self.display_thumbnails_grid(top_videos)
    
    #display the thumbnails for the searched similar videos
    def display_thumbnails_grid(self, top_videos):
        num_videos = len(top_videos)
        cols = 4
        rows = (num_videos + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i in range(len(axes)):
            if i < num_videos:
                video_id, score = top_videos[i]
                thumbnail_path = os.path.join(self.thumbnail_dir, f"{video_id}.jpg")
                img = mpimg.imread(thumbnail_path)
                
                axes[i].imshow(img)
                axes[i].set_title(f"Video {video_id}\nSimilarity: {score:.4f}")
                axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    db_path = "../Database/Phase_3.db"
    latent_model = int(input("Please provide the input for 1 - Layer3 + PCA, 2 - Avgpool + SVD, 3 - KMeans + HOG: "))
    #feature_column = int(input("Enter the feature column to use: "))
    thumbnail_dir = "database/thumbnails"
    num_layers = 3
    hashes_per_layer = 5
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    w = 10
    if latent_model == 1:
        feature_column = 1
        video_search = VideoSearchTool(db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir, "PCA", Feature_Space_Map)
    elif latent_model == 2:
        feature_column = 3
        video_search = VideoSearchTool(db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir, "SVD", Feature_Space_Map)
    elif latent_model==3:
        feature_column = 4
        video_search = VideoSearchTool(db_path, feature_column, num_layers, hashes_per_layer, w, thumbnail_dir, "KMeans", Feature_Space_Map)
    query_video_id = int(input("Enter the query videoID: "))
    t = int(input("Enter the number of similar videos to retrieve: "))
    
    video_search.search(query_video_id, t)

if __name__ == "__main__":
    main()