import numpy as np

# data = U Σ V^T
def SVD(data, latent_count):

    # D^T D
    DtD = np.dot(data.T, data)

    # Calculate eigenvalues and eigenvectors for D^T D
    eigenvalues_V, V = np.linalg.eigh(DtD)

    # Sort the eigenvalues to get the top latent semantics
    sorted_indices = np.argsort(eigenvalues_V)[::-1]
    eigenvalues_V = eigenvalues_V[sorted_indices]
    V = V[:, sorted_indices]
    
    V_subset = V[:, :latent_count]

    # Data in Reduced Dimensional space
    svd_data = np.dot(data, V_subset)

    return svd_data
