# Import the libraries
import numpy as np
import json

from Util.services import ServiceClass as Service


# Calculate the significant eigen values using the mean
def mean_eigen_value(eigenvalues):
    # Calculate the mean
    mean = np.mean(eigenvalues)

    inherent_dim = 0
    # Pick the eigen values that are greater than or equal to mean
    for eigenvalue in eigenvalues:
        if eigenvalue >= mean:
            inherent_dim += 1
        else:
            break
    return inherent_dim

# Function to caluclates the eigen values for the feature space
def get_inherent_dimensionality(feature_space, action, inherent_dimensionality_map):
    query = f"SELECT {Service.all_feature_space_map[feature_space]} FROM data WHERE All_Label == '{action}';"

    Service.c.execute(query)
    rows = Service.c.fetchall()

    # Retrieve all the features
    # Make a list of features matrix
    cleaned_data = []
    if feature_space in [1, 2, 3]:
        for row in rows:
            cleaned_data.append(list(map(float, row[0].strip("[]").split(","))))
    elif feature_space in [4, 5]:
        for row in rows:
            cleaned_data.append(list(map(int, row[0].strip("[]").split())))
    
    max_len = max(len(lst) for lst in cleaned_data)
    # Pad the data if necessary
    padded_data = [lst + [0] * (max_len - len(lst)) for lst in cleaned_data]
    data = np.array(padded_data)

    # Gather the eigen values of the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = eigenvalues[::-1]
    ans = mean_eigen_value(eigenvalues)
    
    print(f"The Inherent Dimensionality for {action} element is: ", ans)
    inherent_dimensionality_map[action] = ans


# Main function to gather the features from the user.
def main():
    feature_space = int(input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF : "))

    inherent_dimensionality_map = {}
    # For each of the target labels, gather the features
    for label in Service.target_labels:
       get_inherent_dimensionality(feature_space, label, inherent_dimensionality_map)

    # Save maximum inherent dimensionality to use when looking at entire Feature Space
    inherent_dimensionality_map['max'] = max(inherent_dimensionality_map.values())

    with open(f'../Outputs/inherent_dim_map_{Service.all_feature_space_map[feature_space]}.json', 'w') as json_file:
        json.dump(inherent_dimensionality_map, json_file)


main()
