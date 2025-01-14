import numpy as np
from sklearn.random_projection import GaussianRandomProjection

def create_hyperplanes_from_data(vectors, num_layers, hashes_per_layer):
    vectors = np.array(vectors)
    lsh_index = {layer: {} for layer in range(num_layers)}

    for layer in range(num_layers):
        # Generate random hyperplanes using GaussianRandomProjection
        rp = GaussianRandomProjection(n_components=hashes_per_layer)
        rp.fit(vectors)
        hyperplanes = rp.components_ 
        print(f"Layer {layer + 1} Hyperplanes (Random Projections):")
        for i, hyperplane in enumerate(hyperplanes):
            print(f"Hyperplane {i + 1}: {hyperplane}")

        # Compute dot products to estimate range
        dot_products = np.dot(vectors, hyperplanes.T)
        dot_min, dot_max = dot_products.min(), dot_products.max()
        print(f"Layer {layer + 1} Dot Product Range: Min={dot_min}, Max={dot_max}")

        # Set 'w' dynamically as a fraction of the range
        w = (dot_max - dot_min) / 10  # Example: divide range into 10 buckets
        print(f"Layer {layer + 1} Selected w: {w}")

        # Generate random biases
        biases = np.random.uniform(dot_min, dot_max, size=hashes_per_layer)
        print(f"Layer {layer + 1} Random Biases: {biases}")

        for idx, vector in enumerate(vectors):
            hash_key = tuple(
                # Hash formula using floor to quantize the projection values into buckets
                int(np.floor((np.dot(vector, hyperplane) + bias) / w))  # Applying floor
                for hyperplane, bias in zip(hyperplanes, biases)
            )
            
            # Ensure the hash values fall within the range [0, 9] by clamping
            hash_key = tuple(max(0, min(9, val)) for val in hash_key)

            if hash_key not in lsh_index[layer]:
                lsh_index[layer][hash_key] = []
            lsh_index[layer][hash_key].append(idx)

    return lsh_index

def main():
    num_layers = int(input("Enter the number of layers: "))
    hashes_per_layer = int(input("Enter the number of hashes per layer: "))
    
    num_vectors = int(input("Enter the number of vectors: "))
    dim = int(input("Enter the dimensionality of each vector: "))
    
    vectors = []
    print("Enter each vector as space-separated values:")
    for i in range(num_vectors):
        vector = list(map(float, input(f"Vector {i + 1}: ").strip().split()))
        if len(vector) != dim:
            print(f"Error: Vector {i + 1} must have {dim} dimensions.")
            return
        vectors.append(vector)

    lsh_index = create_hyperplanes_from_data(vectors, num_layers, hashes_per_layer)

    for layer in range(num_layers):
        print(f"\nLayer {layer + 1} hash index:")
        for hash_key, indices in lsh_index[layer].items():
            print(f"Hash Key: {hash_key} -> Vectors: {indices}")

if __name__ == "__main__":
    main()
