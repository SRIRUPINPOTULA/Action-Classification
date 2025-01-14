import csv
import json
import numpy as np

# data = U Σ V^T
def SVD(data, latent_count, feature_space):

    # D^T D
    DtD = np.dot(data.T, data)

    # Calculate eigenvalues and eigenvectors for D^T D
    eigenvalues_V, V = np.linalg.eigh(DtD)

    # Sort the eigenvalues to get the top latent semantics
    sorted_indices = np.argsort(eigenvalues_V)[::-1]
    eigenvalues_V = eigenvalues_V[sorted_indices]
    V = V[:, sorted_indices]
    
    V_subset = V[:, :latent_count]

    # Calculate singular values (square roots of eigenvalues of D^T D)
    singular_values = np.sqrt(eigenvalues_V)

    # D D^T
    DDt = np.dot(data, data.T)

    # Calculate eigenvalues and eigenvectors of D D^T
    eigenvalues_U, U = np.linalg.eigh(DDt)

    # Sort the eigenvalues of U
    sorted_indices_U = np.argsort(eigenvalues_U)[::-1]
    eigenvalues_U = eigenvalues_U[sorted_indices_U]
    U = U[:, sorted_indices_U]

    U_subset = U[:, :latent_count]

    # Core matrix
    Sigma = np.zeros((latent_count, latent_count), dtype=float)
    np.fill_diagonal(Sigma, singular_values[:latent_count])

    # Data in Reduced Dimensional space
    svd_data = np.dot(np.dot(data, V_subset), Sigma)

    # videoID-wieght pairs
    svd_data_json = {}
    for index, data_row in enumerate(svd_data):
        svd_data_json[index*2] = data_row.tolist()

    print(f"Top-{latent_count} latent Semantics for SVD")
    for index, eigenvalue in enumerate(singular_values[:latent_count]):
        print(f"{index} - {eigenvalue}")
    
    with open('../Outputs/Task_2/SVD_left_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(U_subset)
    
    with open('../Outputs/Task_2/SVD_core_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Sigma)

    with open('../Outputs/Task_2/SVD_right_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(V_subset.T)

    np.save('../Outputs/Task_2/SVD_right_matrix.npy', V_subset)

    with open('../Outputs/Task_2/SVD_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(svd_data)

    with open(f'../Outputs/Task_2/videoID-weight_SVD_{feature_space}.json', 'w') as f:
        json.dump(svd_data_json, f, indent=4)
