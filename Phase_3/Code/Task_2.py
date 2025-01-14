import json
from Util.KMeanslatentmodel import KMeans_implementation
from Util.KMeanslatentmodel import PCA
from Util.KMeanslatentmodel import SVD
import sqlite3
import csv
import numpy as np

connection = sqlite3.connect('../Database/Phase_3.db')
c = connection.cursor()

target_label_range = {}

#List of target video labels
target_videos = ['golf',  'shoot_ball', 'brush_hair', 'handstand', 'shoot_bow', 
                'cartwheel', 'hit', 'shoot_gun', 'hug', 'sit', 'catch', 
                'jump', 'situp', 'chew', 'kick', 'smile', 'clap', 'kick_ball', 'smoke',
                'climb', 'somersault', 'climb_stairs', 'laugh', 'stand']

#List of video labels
video_labels = [
    "stand", "smoke", "sit", "shoot_gun", "brush_hair", "clap", "climb_stairs", 
    "shoot_bow", "hit", "shoot_ball", "jump", "handstand", "kick", "situp", 
    "smile", "catch", "golf", "chew", "cartwheel", "laugh", "hug", "somersault", 
    "climb", "kick_ball", "run", "fencing", "shake_hands", "walk", "draw_sword", 
    "sword_exercise", "eat", "pour", "ride_horse", "push", "flic_flac", "pick", 
    "sword", "drink", "pushup", "talk", "swing_baseball", "turn", "pullup", 
    "throw", "fall_floor", "kiss", "dribble", "ride_bike", "punch", "dive", "wave"
]

#Train the binary svm classification
def train_binary_svm(X, y, learning_rate=0.001, lambda_param=0.01, epochs=1000):
    
    if X.ndim == 1: 
        X = X.reshape(-1, 1)
    #Gather the shape of feature matrix X
    n_samples, n_features = X.shape
    #Initialise the weight and bias as zero
    w = np.zeros(n_features)
    b = 0
    #Iterate through all the epochs
    for epoch in range(epochs):
        #Update the weight and bias
        for idx, x_i in enumerate(X):
            condition = y[idx] * (np.dot(x_i, w) + b) >= 1
            if condition:
                dw = 2 * lambda_param * w
                db = 0
            else:
                dw = 2 * lambda_param * w - y[idx] * x_i
                db = -y[idx]
            w -= learning_rate * dw
            b -= learning_rate * db
    #return the weight and bias
    return w, b

#Training the binary svm for all the classes
def train_one_vs_rest_svm(X, y, learning_rate=0.001, lambda_param=0.01, epochs=1000):
    #For each unique label call train binary svm
    classes = np.unique(y)
    models = {}
    for c in classes:
        binary_y = np.where(y == c, 1, -1)  
        w, b = train_binary_svm(X, binary_y, learning_rate, lambda_param, epochs)
        #Update the weight anad red
        models[c] = (w, b)
    #For each of the classes gathered weight and bias are returned.
    return models

#Predict the class for the query feature
def predict_one_vs_rest_svm(X, models, m):
    if X.ndim == 1: 
        X = X.reshape(1, -1)
    
    #Calculate the score for each scores
    scores = {c: np.dot(X, models[c][0]) + models[c][1] for c in models}

    predictions = []
    for i in range(X.shape[0]):
        #Gather the top m classes
        sample_scores = {c: scores[c][i] for c in models}
        top_m_classes = sorted(sample_scores, key=sample_scores.get, reverse=True)[:m]
        #Append the top m classes to the predictions
        predictions.append(top_m_classes)
    #Return the classes
    return predictions if len(predictions) > 1 else predictions[0]

#Define the PCA function for the latent model adn query feature
def PCA(feature_space, query_feature, classifier, k, m):
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    cleaned_data = []
    labels_predicted = []
    initial = 0
    
    for i in range(0, len(video_labels)):
        action = video_labels[i]
        retrieval_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE All_Label == '{action}';"
        c.execute(retrieval_query)
        rows = c.fetchall()
        for row in rows:
            cleaned_data.append(list(map(float, row[0].strip("[]").split(","))))
        target_label_range[action]= [initial, len(cleaned_data)-1]
        initial = len(cleaned_data)
        
    data = np.array(cleaned_data)
    matches = np.all(data == query_feature, axis=1)
    indices = np.where(matches)[0]
    ##Latent Count
    latent_count = 50
    
    ##Gathered the feature model representation
    row, column = data.shape
    
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
    #Before returing the function
    query_feature = pca_data[indices[0]]
    
    #KNN_Classifier to predict k similar labels
    if classifier==1:
        labels_predicted = KNN_classifier(pca_data, target_label_range, query_feature, k, m)
    #Implementation of SVM
    else:
        Y = []
        #Make a list of Y that has the all the video labels
        for key, item in target_label_range.items():
            for i in range(item[0], item[1] + 1):
                Y.append(key)
        #Delete the entry where the query feature matches
        pca_data = np.delete(pca_data, indices, axis = 0)
        del Y[indices[0]]
        Y = np.array(Y).flatten()
        #Train the SVM 
        models = train_one_vs_rest_svm(pca_data, Y, learning_rate=0.001, lambda_param=0.01, epochs=100)
        #Predict the class for the query
        predicted_class = predict_one_vs_rest_svm(query_feature, models, m)
        predicted_class = [str(cls) for cls in predicted_class]
        for i in range(len(predicted_class)):
            print("The predicted m labels are: ", predicted_class[i])
            labels_predicted.append(predicted_class[i])
    return labels_predicted
            
##Define the SVD function for the latent model adn query feature
def SVD(feature_space, query_feature, classifier, k, m):
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    
    cleaned_data = []
    labels_predicted = []
    initial = 0
    
    for i in range(0, len(video_labels)):
        action = video_labels[i]
        retrieval_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE All_Label == '{action}';"
        c.execute(retrieval_query)
        rows = c.fetchall()
        for row in rows:
            cleaned_data.append(list(map(float, row[0].strip("[]").split(","))))

        target_label_range[action]= [initial, len(cleaned_data)-1]
        initial = len(cleaned_data)

    data = np.array(cleaned_data)
    matches = np.all(data == query_feature, axis=1)
    indices = np.where(matches)[0]
    # D^T D
    DtD = np.dot(data.T, data)

    # Calculate eigenvalues and eigenvectors for D^T D
    eigenvalues_V, V = np.linalg.eigh(DtD)

    #latent count
    latent_count = 76
    
    # Sort the eigenvalues to get the top latent semantics
    sorted_indices = np.argsort(eigenvalues_V)[::-1]
    eigenvalues_V = eigenvalues_V[sorted_indices]
    V = V[:, sorted_indices]
    
    V_subset = V[:, :latent_count]

    # Data in Reduced Dimensional space
    svd_data = np.dot(data, V_subset)
    query_feature = svd_data[indices[0]]
    #Select the classifier
    if classifier==1:
        labels_predicted = KNN_classifier(svd_data, target_label_range, query_feature, k, m)
    else:
        Y = []
        for key, item in target_label_range.items():
            for i in range(item[0], item[1] + 1):
                Y.append(key)
        svd_data = np.delete(svd_data, indices, axis = 0)
        del Y[indices[0]]
        Y = np.array(Y).flatten()
        #Train the SVM
        models = train_one_vs_rest_svm(svd_data, Y, learning_rate=0.001, lambda_param=0.01, epochs=100)
        #Predict the class for the query features
        predicted_class = predict_one_vs_rest_svm(query_feature, models, m)
        predicted_class = [str(cls) for cls in predicted_class]
        for i in range(len(predicted_class)):
            print("The predicted m labels are: ", predicted_class[i])
            labels_predicted.append(predicted_class[i])
    return labels_predicted

#KNN Classifier
def KNN_classifier(data, target_label_range, query_feature, k, m):
    data = np.array(data)
    query_feature = np.array(query_feature)
    #Using L2 norm find the distance between the query feature and the data
    distances = np.linalg.norm(data - query_feature, axis=1)
    #Pick the k most significant clusters
    closest_indices = np.argsort(distances)[:k+1]
    c = closest_indices.tolist()
    labels_predicted = []
    for i in range(1, len(c)):
        index = c[i]
        #Append the labels for the closest indices selected
        for key, item in target_label_range.items():  
            if item[0]<=index  and item[1]>=index:
                labels_predicted.append(key)
                break
    #If the value of m is less than k only k labels can be predicted
    if m>k:
        print("The value of k is less than m. K labels would be predicted. ")
        m = min(m, k)
    #Predict m labels
    for i in range(0, m):
        print("The predicted m labels are: ", labels_predicted[i])
    return labels_predicted

#Implementaiton of KMeans
def gather_features(feature_space, query_feature, classifier, k, m):
    #For the feature space gather all the feature
    Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF"}
    cleaned_data = []
    initial = 0
    for i in range(0, len(video_labels)):
        #Gather all the video for the video labels
        action = video_labels[i]
        retrieval_query = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE All_Label == '{action}';"
        c.execute(retrieval_query)
        rows = c.fetchall()
        for row in rows:
            cleaned_data.append(list(map(int, row[0].strip("[]").split())))
        target_label_range[action]= [initial, len(cleaned_data)-1]
        initial = len(cleaned_data)
    #If the size is less than 480 then append zero.
    max_len = max(len(lst) for lst in cleaned_data)
    padded_data = [lst + [0] * (max_len - len(lst)) for lst in cleaned_data]
    data = np.array(padded_data)
    matches = np.all(data == query_feature, axis=1)
    indices = np.where(matches)[0]
    indices = indices.astype(int).flatten()
    #Gather the latent model using KMeans
    latent_model_kmeans = KMeans_implementation(data, 87)
    #Gather the query feature
    query_feature = latent_model_kmeans[indices[0]]
    query_feature = np.array(query_feature)
    #Call the KNN Classifier and SVM Classifier based on user query
    if classifier==1:
        predicted_class = KNN_classifier(latent_model_kmeans, target_label_range, query_feature, k, m)
    else:
        Y = []
        for key, item in target_label_range.items():
            for i in range(item[0], item[1] + 1):
                Y.append(key)
        del latent_model_kmeans[indices[0]]
        Y = np.array(Y).flatten()
        latent_model_kmeans = np.array(latent_model_kmeans)
        #Train the svm based on labels
        models = train_one_vs_rest_svm(latent_model_kmeans, Y, learning_rate=0.001, lambda_param=0.01, epochs=100)
        #Predict the class for the query feature
        predicted_class = predict_one_vs_rest_svm(query_feature, models, m)
        predicted_class = [str(cls) for cls in predicted_class]
        for i in range(len(predicted_class)):
            print("The predicted m labels are: ", predicted_class[i])
    return predicted_class

#main function that gatheres the latent model and videoID
def main():
    #Gather the videoID
    video_number = int(input("Provide the  VideoID: "))
    #If the videoID is even end the execution
    if video_number%2==0:
        print("Please provide odd number for video id.")
        return
    #Provide the classifier KNN, SVM
    classifier = int(input("Provide the  classifier 1 - KNN, 2 - SVM: "))
    #Gather the latent model
    latent_model = int(input("Please provide the input for 1 - Layer3 + PCA, 2 - Avgpool + SVD, 3 - KMeans + HOG: "))
    #Value of m to predict target label
    m = int(input("Provide the value for m : "))
    #If classifier is knn gather the value of k
    if classifier==1:
        k = int(input("Provide the value for k : "))
    else:
        k = 0
    #Depending on latent model predict m labels
    if latent_model == 3:
        Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF: "}
        feature_space = 4
        #Gather the feature for the given videoID
        query_feature = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE videoID = {video_number};"
        c.execute(query_feature)
        result = c.fetchone()
        result = result[0]
        cleaned_data = result.replace('[', '').replace(']', '').replace('\n', '').strip()
        query_feature = list(map(int, cleaned_data.split()))
        if len(query_feature) < 480:
            zeros_to_add = 480 - len(query_feature)
            query_feature.extend([0] * zeros_to_add)
        results = gather_features(feature_space, query_feature, classifier, k, m)
        true_label = f"SELECT All_label FROM data WHERE videoID = {video_number};"
        c.execute(true_label)
        result = c.fetchone()
        result = result[0]
        if result in target_videos:
            if result in results:
                index = results.index(result)
                formula = (m - index + 1 + 1)/m 
                print("Per-classifier accuracy value is: ", formula)
            else:
                print("The per-classifier accuracy value is zero")
    elif latent_model == 2:
        Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF: "}
        feature_space = 3
        #Gather the feature for the given videoID
        query_feature = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE videoID = {video_number};"
        c.execute(query_feature)
        result = c.fetchone()
        result = result[0]
        cleaned_data = result.replace('[', '').replace(']', '').replace('\n', '').strip()
        query_feature = list(map(float, cleaned_data.strip("[]").split(",")))
        results = SVD(feature_space, query_feature, classifier, k, m)
        true_label = f"SELECT All_label FROM data WHERE videoID = {video_number};"
        c.execute(true_label)
        result = c.fetchone()
        result = result[0]
        if result in target_videos:
            if result in results:
                index = results.index(result)
                formula = (m - index + 1 + 1)/m 
                print("Per-classifier accuracy value is: ", formula)
            else:
                print("The per-classifier accuracy value is zero")
    else:
        Feature_Space_Map = {1: "Layer_3", 2: "Layer_4", 3: "AvgPool", 4: "BOF_HOG", 5: "BOF_HOF: "}
        feature_space = 1
        #Gather the feature for the given videoID
        query_feature = f"SELECT {Feature_Space_Map[feature_space]} FROM data WHERE videoID = {video_number};"
        c.execute(query_feature)
        result = c.fetchone()
        result = result[0]
        cleaned_data = result.replace('[', '').replace(']', '').replace('\n', '').strip()
        query_feature = list(map(float, cleaned_data.strip("[]").split(",")))
        results = PCA(feature_space, query_feature, classifier, k, m)
        true_label = f"SELECT All_label FROM data WHERE videoID = {video_number};"
        c.execute(true_label)
        result = c.fetchone()
        result = result[0]
        if result in target_videos:
            if result in results:
                index = results.index(result)
                formula = (m - index + 1 + 1)/m 
                print("Per-classifier accuracy value is: ", formula)
            else:
                print("The per-classifier accuracy value is zero")
main()