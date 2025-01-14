import os
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)


#function to create label-label similarity matrix
def label_similarity_matrix(feature_model_data, labels):
    label_set = list(set(labels.values()))
    n_labels = len(label_set)
    
    similarity_matrix = np.zeros((n_labels, n_labels))
    
    for i, label1 in enumerate(label_set):
        for j, label2 in enumerate(label_set):
            if i <= j:
                vids1 = [vid for vid, lab in labels.items() if lab == label1 and int(vid[3:]) % 2 == 0]
                vids2 = [vid for vid, lab in labels.items() if lab == label2 and int(vid[3:]) % 2 == 0]
                
                if vids1 and vids2:
                    avg_vec1 = np.mean([feature_model_data[vid] for vid in vids1], axis=0)
                    avg_vec2 = np.mean([feature_model_data[vid] for vid in vids2], axis=0)
                    
                    similarity = 1 - cosine(avg_vec1, avg_vec2)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
    
    return similarity_matrix, label_set

#function to perform dimensionality reduction
def dimensionality_reduction(similarity_matrix, method='PCA', components=2):
    if method == 'PCA':
        model = PCA(n_components=components)
    elif method == 'SVD':
        model = TruncatedSVD(n_components=components)
    elif method == 'LDA':
        model = LDA(n_components=components)
    elif method == 'kmeans':
        model = KMeans(n_clusters=components)
    
    reduced_matrix = model.fit_transform(similarity_matrix)
    return reduced_matrix, model

#main function
def task_5(feature_model_data, labels, s, method='PCA'):
    
    similarity_matrix, label_set = label_similarity_matrix(feature_model_data, labels)
    reduced_matrix, model = dimensionality_reduction(similarity_matrix, method=method, components=s)
    output_file = os.path.join('output', 'latent_semantics.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({'reduced_matrix': reduced_matrix, 'model': model}, f)
    weights = np.sum(reduced_matrix, axis=0)
    label_weight_pairs = sorted(list(zip(label_set, weights)), key=lambda x: x[1], reverse=True)
    
    output_label_weights = os.path.join('output', 'label_weights.txt')
    with open(output_label_weights, 'w') as f:
        f.write("Label-Weight pairs (in decreasing order of weights):\n")
        for label, weight in label_weight_pairs:
            f.write(f"Label: {label}, Weight: {weight}\n")
    
    
    print(f"Latent semantics and label-weight pairs stored in the 'output' folder.")
    
    return reduced_matrix, label_weight_pairs

#function to load feature model data
def load_feature_model(feature_model_name):
    #dummy data for trial
    feature_model_data = {
        'vid1': np.random.rand(512),
        'vid2': np.random.rand(512),
        'vid3': np.random.rand(512),
        'vid4': np.random.rand(512)
    }

    labels = {
        'vid1': 'golf',
        'vid2': 'shoot ball',
        'vid3': 'brush hair',
        'vid4': 'handstand'
    }
    
    return feature_model_data, labels


def main():
    feature_model_name = input("Enter the feature model (e.g., R3D18-Layer3-512, BOF-HOG-480, etc.): ")
    feature_model_data, labels = load_feature_model(feature_model_name)
    s = int(input("Enter the number of components for dimensionality reduction (s): "))
    print("Select the dimensionality reduction method:")
    print("1. PCA")
    print("2. SVD")
    print("3. LDA")
    print("4. k-means")
    
    method_input = input("Enter the number corresponding to the method: ")
    if method_input == '1':
        method = 'PCA'
    elif method_input == '2':
        method = 'SVD'
    elif method_input == '3':
        method = 'LDA'
    elif method_input == '4':
        method = 'kmeans'
    else:
        print("Invalid input. Defaulting to PCA.")
        method = 'PCA'
    
    task_5(feature_model_data, labels, s, method)

if __name__ == "__main__":
    main()