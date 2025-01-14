import csv
import json
import numpy as np

def PCA(data, latent_count, feature_space):

    # We use covariance matrix for PCA
    cov_matrix = np.cov(data, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues to get the top latent semantics
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    eigenvalues_subset = sorted_eigenvalues[:latent_count]
    eigenvectors_subset = sorted_eigenvectors[:, :latent_count]

    # Left factor matrix: Principal components (eigenvectors)
    left_matrix = eigenvectors_subset
    # Right factor matrix: Principal components (eigenvectors) ^ Transpose
    right_matrix = eigenvectors_subset.T
    # Core matrix: Diagnal matrix of Eigenvalues
    core_matrix = np.zeros((latent_count, latent_count), dtype=float)
    np.fill_diagonal(core_matrix, sorted_eigenvalues[:latent_count])
    
    # Data in Reduced Dimensional space
    pca_data = np.dot(data, eigenvectors_subset)

    # videoID-wieght pairs
    pca_data_json = {}
    for index, data_row in enumerate(pca_data):
        pca_data_json[index*2] = data_row.tolist()

    print(f"Top-{latent_count} latent Semantics for PCA")
    for index, eigenvalue in enumerate(eigenvalues_subset):
        print(f"{index} - {eigenvalue}")

    with open('../Outputs/Task_2/PCA_left_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(left_matrix)

    np.save('../Outputs/Task_2/PCA_left_matrix.npy', left_matrix)
    
    with open('../Outputs/Task_2/PCA_core_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(core_matrix)
    
    with open(f'../Outputs/Task_2/videoID-weight_PCA_{feature_space}.json', 'w') as f:
        json.dump(pca_data_json, f, indent=4)
