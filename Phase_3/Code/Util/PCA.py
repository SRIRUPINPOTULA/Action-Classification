import numpy as np

def PCA(data, latent_count):

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
    
    # Data in Reduced Dimensional space
    pca_data = np.dot(data, left_matrix)

    return pca_data
