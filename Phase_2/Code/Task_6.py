import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans


def load_data():
    #replace this with actual loading logic from Task 0, 2, or 4
    video_features = {
        'label1': np.random.rand(512),
        'label2': np.random.rand(512),
        'label3': np.random.rand(512),
        'label4': np.random.rand(512)
    }
    return video_features

def apply_dimensionality_reduction(model, data, technique, s):
    if technique == 'PCA':
        pca = PCA(n_components=s)
        return pca.fit_transform(data)
    elif technique == 'SVD':
        svd = TruncatedSVD(n_components=s)
        return svd.fit_transform(data)
    elif technique == 'LDA':
        lda = LDA(n_components=s)
        return lda.fit_transform(data)
    elif technique == 'k-means':
        kmeans = KMeans(n_clusters=s)
        return kmeans.fit_transform(data)
    else:
        raise ValueError(f"Unknown dimensionality reduction technique: {technique}")

def find_similar_labels(target_label, feature_model, l):
    target_vector = feature_model[target_label]
    
    #compute similarities using cosine similarity
    label_vectors = np.array(list(feature_model.values()))
    labels = list(feature_model.keys())
    
    similarities = cosine_similarity([target_vector], label_vectors)[0]
    
    similar_indices = np.argsort(similarities)[-l:][::-1]
    similar_labels = [(labels[i], similarities[i]) for i in similar_indices]
    
    return similar_labels

if __name__ == "__main__":
    feature_model = load_data()

    target_label = input("Enter target or non-target label: ")
    feature_or_latent = input("Enter feature model or latent semantics (Task 0/2/4): ")
    l = int(input("Enter the number of similar labels to retrieve: "))

    
    technique = input("Enter dimensionality reduction technique (PCA/SVD/LDA/k-means) or skip: ").lower()
    s = int(input("Enter number of dimensions to reduce to (if applicable): ")) if technique else None

    if technique:
        feature_model = apply_dimensionality_reduction(feature_or_latent, np.array(list(feature_model.values())), technique, s)
    
    similar_labels = find_similar_labels(target_label, feature_model, l)
    
    print(f"\nTop {l} most similar labels to '{target_label}':")
    for label, score in similar_labels:
        print(f"{label}: {score:.4f}")