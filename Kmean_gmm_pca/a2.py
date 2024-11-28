# Statistical Methods in Artificial Intelligence
#  Assignment 2


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import sys
sys.path.append("../..")
from models.gmm.gmm import GMM
from models.pca.pca import PCA
from models.k_means.k_means import KMeans
from models.knn.knn import Knn
from performance_measures.accuracy import accuracy_score
from performance_measures.f1_score import f1_score
from performance_measures.precision import precision_score
from performance_measures.recall import recall_score


#  Dimensionality Reduction and Visualization
# 5.2 Perform Dimensionality Reduction

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load feather file
data = pd.read_feather('../../data/external/word.feather')
words = data['words'].values
embeddings = np.array(data['vit'].tolist())
# Instantiate PCA for 2D and 3D
pca_2d = PCA(n_components=2)
pca_2d.fit(embeddings)
embeddings_2d = pca_2d.transform(embeddings)
pca_3d = PCA(n_components=3)
pca_3d.fit(embeddings)
embeddings_3d = pca_3d.transform(embeddings)
print("2D PCA Check:", pca_2d.checkPCA(embeddings))
print("3D PCA Check:", pca_3d.checkPCA(embeddings))
# 2D Plot
plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
plt.title('PCA 2D Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], alpha=0.7)
ax.set_title('PCA 3D Projection')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()





# 3. K-Means Clustering 

def load_data(filepath):
    df = pd.read_feather(filepath)
    words = df['words'].values
    embeddings = np.vstack(df['vit'].values)
    return words, embeddings

def determine_optimal_k(X):
    wcss = []
    k_range = range(1, 20)  
    for k in k_range:
        kmeans = KMeans(k=k)
        kmeans.fit(X)
        wcss.append(kmeans.getCost())
    
    plt.plot(k_range, wcss)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    return wcss

def find_optimal_k(wcss):
    first_derivative = np.diff(wcss)
    second_derivative = np.diff(first_derivative)
    # Find the index of the maximum second derivative, as the "elbow" occurs at that point
    optimal_k = np.argmax(second_derivative) + 2  # +2 because np.diff reduces dimensionality by 1 twice
    return optimal_k


#  3.2 Determining the Optimal Number of Clusters for 512 dimensions
# Load feather file

if __name__ == "__main__":
    filepath = '../../data/external/word.feather'
    words, embeddings = load_data(filepath)
    
    wcss = determine_optimal_k(embeddings)

# Varying the value of k and plot the Within-Cluster
#  Sum of Squares (WCSS) against k to identify the ”elbow” point, which
#  indicates the optimal number of clusters. We’ll refer to this as kkmeans1

kmeans1 = KMeans(k=9)
kmeans1.fit(embeddings)
# Get cluster assignments
clusters = kmeans1.predict(embeddings)
print("Cluster assignments:", clusters)
print("Cost (WCSS):", kmeans1.getCost())
#  Performing K-means clustering on the dataset using the number of clusters
#  as kmeans1 

# Assign cluster labels back to the words
clustered_df = pd.DataFrame({
    'Word': words,
    'Cluster': clusters
})
clustered_df_sorted = clustered_df.sort_values(by='Cluster')
# Save the results to .csv file
clustered_df_sorted.to_csv('clustered_words.csv', index=False)


# 4 Gaussian Mixture Models 

# 4.2 Determining the Optimal Number of Clusters for 512 dimensions

# Perform GMM clustering on the give dataset for any number of clusters.

# Load feather file
df = pd.read_feather('../../data/external/word.feather')
X = np.array(df['vit'].tolist())

# Range of components to test
max_components = 10

# Custom GMM evaluation
def evaluate_custom_gmm(X, max_components):
    best_aic = np.inf
    best_bic = np.inf
    best_aic_components = 0
    best_bic_components = 0

    for n_components in range(1, max_components + 1):
        gmm = GMM(n_components=n_components)
        gmm.fit(X)
        aic, bic = gmm.calculate_aic_bic(X)

        print(f"Custom GMM - Components: {n_components}, AIC: {aic}, BIC: {bic}")

        if aic < best_aic:
            best_aic = aic
            best_aic_components = n_components
        
        if bic < best_bic:
            best_bic = bic
            best_bic_components = n_components

    print(f"Optimal number of components based on AIC: {best_aic_components}")
    print(f"Optimal number of components based on BIC: {best_bic_components}")

evaluate_custom_gmm(X, max_components)

# Now performing GMM clustering using the
# #  sklearn GMM class. 

def evaluate_sklearn_gmm(X, max_components):
    best_aic = np.inf
    best_bic = np.inf
    best_aic_components = 0
    best_bic_components = 0

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(X)
        aic = gmm.aic(X)
        bic = gmm.bic(X)

        print(f"Scikit-learn GMM - Components: {n_components}, AIC: {aic}, BIC: {bic}")

        if aic < best_aic:
            best_aic = aic
            best_aic_components = n_components
        
        if bic < best_bic:
            best_bic = bic
            best_bic_components = n_components

    print(f"Optimal number of components based on AIC: {best_aic_components}")
    print(f"Optimal number of components based on BIC: {best_bic_components}")

evaluate_sklearn_gmm(X, max_components)

# 6.  PCA+Clustering
# 6.1 K-means Clustering Based on 2D Visualization

# Assume kmeans1 is the optimal number of clusters found from the elbow method
kmeans1 = 3 
# Perform K-means clustering
kmeans = KMeans(k=kmeans1)
kmeans.fit(embeddings_2d)
clusters = kmeans.predict(embeddings_2d)
clustered_df = pd.DataFrame({
    'Word': words,
    'Cluster': clusters
})
# Save the results 
clustered_df_sorted = clustered_df.sort_values(by='Cluster')
clustered_df_sorted.to_csv('clustered_words_kmeans1.csv', index=False)
def plot_clusters(embeddings_2d, clusters, words, title="2D Word Clusters"):
    # Create a scatter plot of the embeddings
    plt.figure(figsize=(10, 8))
    
    # Plot each cluster with a different color
    for cluster in np.unique(clusters):
        cluster_points = embeddings_2d[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', s=50)

    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.75)

    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_clusters(embeddings_2d, clusters, words)
# We take k2 from the 2D visualization
k2 = 3 # Number of clusters estimated from 2D plot
kmeans_2d = KMeans(k=k2)
kmeans_2d.fit(embeddings_2d)
clusters = kmeans_2d.predict(embeddings_2d)
clustered_df = pd.DataFrame({
    'Word': words,
    'Cluster': clusters
})
# Save the results 
clustered_df_sorted = clustered_df.sort_values(by='Cluster')
clustered_df_sorted.to_csv('clustered_words_kmean_k2.csv', index=False)


# 6.2 PCA + K-Means Clustering

# PLOTTING SCREE PLOT
pca = PCA(n_components=512)
pca.fit(embeddings)
total_variance = np.sum(pca.explained_variance_)
explained_variance_ratio = (pca.explained_variance_ / total_variance) * 100  # Convert to percentage
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
threshold = 94
components_80_pct = np.argmax(cumulative_explained_variance_ratio >= threshold) + 2  # +1 for indexing
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(1, 513)), 
    y=cumulative_explained_variance_ratio,
    mode='lines+markers',
    marker=dict(size=6),
    line=dict(dash='dash'),
    hovertemplate='<b>Component %{x}</b><br>Cumulative Explained Variance: %{y:.2f}%',
    name='Explained Variance'
))
fig.add_trace(go.Scatter(
    x=[components_80_pct],
    y=[cumulative_explained_variance_ratio[components_80_pct - 1]],  # -1 because of Python indexing
    mode='markers+text',
    text=[f'90% variance at {components_80_pct} PCs'],
    textposition='top right',
    marker=dict(size=12, color='red'),
    name='80% Variance Threshold'
))
fig.update_layout(
    title="Scree Plot with Cumulative Explained Variance (%)",
    xaxis_title="Number of Principal Components",
    yaxis_title="Cumulative Explained Variance (%)",
    template="plotly_white",
    hovermode="closest",
    showlegend=False,
    width=800,   
    height=600   
)
fig.show()
# GOT THE DIMENSION
optimal_dims = 150 # Number of dimensions chosen from the scree plot
pca_optimal = PCA(n_components=optimal_dims)
reduced_data = pca_optimal.fit_transform(embeddings)
wcss_r = determine_optimal_k(reduced_data)
# Assuming kmeans1 is the optimal number of clusters found from the elbow method
kmeans3 = 5  
# Perform K-means clustering
kmeans = KMeans(k=kmeans3)
kmeans.fit(reduced_data)
clusters = kmeans.predict(reduced_data)
clustered_df = pd.DataFrame({
    'Word': words,
    'Cluster': clusters
})
clustered_df_sorted = clustered_df.sort_values(by='Cluster')
# Save the results 
clustered_df_sorted.to_csv('clustered_words_reduced_kmeans3.csv', index=False)

#  6.3 GMM Clustering Based on 2D Visualization
gmm_k2 = GMM(n_components=3)
  # Example for 6 clusters
gmm_k2.fit(reduced_data)

#  6.4 PCA + GMMs
# Determining the optimal number of clusters using BIC and AIC
def find_optimal_clusters(data):
    bic_scores = []
    aic_scores = []
    cluster_range = range(1,11)
    
    for n_clusters in cluster_range:
        gmm = GMM(n_components= n_clusters)
        gmm.fit(data)
        aic, bic = gmm.calculate_aic_bic(data)
        bic_scores.append(bic)  # Append only BIC score
        aic_scores.append(aic)  # Append only AIC score
   
    return cluster_range, bic_scores, aic_scores
cluster_range, bic_scores, aic_scores = find_optimal_clusters(reduced_data)
# Plot BIC and AIC scores
plt.figure(figsize=(12, 6))
plt.plot(cluster_range, bic_scores, label='BIC', marker='o', color='blue')
plt.plot(cluster_range, aic_scores, label='AIC', marker='o', color='red')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('BIC and AIC for Different Number of Clusters')
plt.legend()
plt.grid(True)
plt.show()
# from observation
kgmm3 = 5
gmm = GMM(n_components=kgmm3)
gmm.fit(reduced_data)
# After fitting the GMM and obtaining the membership probabilities
membership = gmm.getMembership(reduced_data)
cluster_labels = np.argmax(membership, axis=1)
clustered_df = pd.DataFrame({
    'Word': data['words'],  
    'Cluster': cluster_labels
})
clustered_df_sorted = clustered_df.sort_values(by='Cluster')
clustered_df_sorted.to_csv('clustered_words_gmm_kgmm3.csv', index=False)


# 8. Hierarchical Clustering


# Computing the linkage matrix using Euclidean distance and different linkage methods
Z_single = linkage(reduced_data, method='single', metric='euclidean')
Z_complete = linkage(reduced_data, method='complete', metric='euclidean')
Z_average = linkage(reduced_data, method='average', metric='euclidean')
Z_ward = linkage(reduced_data, method='ward')
# Ploting dendrogram for each linkage method
plt.figure(figsize=(26, 7))
plt.title("Dendrogram (Single Linkage)")
dendrogram(Z_single)
plt.show()
plt.figure(figsize=(25, 7))
plt.title("Dendrogram (Complete Linkage)")
dendrogram(Z_complete)
plt.show()
plt.figure(figsize=(26, 7))
plt.title("Dendrogram (Average Linkage)")
dendrogram(Z_average)
plt.show()
plt.figure(figsize=(26, 7))
plt.title("Dendrogram (Ward's Linkage)")
dendrogram(Z_ward)
plt.show()

kbest1 = 2  # optimal value from K-Means
kbest2 = 5  # optimal value from GMM
# we will get clusters using fcluster by cutting at the number of clusters (for kbest1 and kbest2)
clusters_kbest1 = fcluster(Z_ward, kbest1, criterion='maxclust')
clusters_kbest2 = fcluster(Z_ward, kbest2, criterion='maxclust')
plt.figure(figsize=(30, 7))
plt.title("Dendrogram with Cut Line (Complete Linkage)")
dendrogram(Z_ward)
# The height here corresponds to the level where kbest1 clusters are formed
plt.axhline(y=Z_ward[-kbest1, 2], c='r', linestyle='--', label=f'Cut for kbest1 = {kbest1}')
plt.axhline(y=Z_ward[-kbest2, 2], c='g', linestyle='--', label=f'Cut for kbest2 = {kbest2}')
plt.legend()
plt.show()


# 9.  Nearest Neighbor Search

#  9.1 PCA + KNN


df = pd.read_csv('../../data/external/dataset.csv')
df = df.iloc[:,1:]
df = df.drop_duplicates(subset='track_id')
print(df.shape)
df = df.iloc[:,4:]
df['track_genre'] = pd.factorize(df['track_genre'])[0]
df['explicit'] = pd.factorize(df['explicit'])[0]
df = df.sample(n=1000, random_state=0)
Z = df.iloc[:, 0:16].values
mean = np.mean(Z, axis=0)
std_dev = np.std(Z, axis=0)
# Standardize the data
Z_standardized = (Z - mean) / std_dev
X = Z_standardized[:, :15]  # X will be the standardized values from column 0 to 14
y = Z_standardized[:, -1]   
column_names = df.columns[:15] 
# Convert the standardized array (X and y) back to a DataFrame
df_standardized = pd.DataFrame(Z_standardized, columns=list(column_names) + ['track_genre'])
numerical_columns = df_standardized.select_dtypes(include=['number']).columns
# Extract numerical data
newly_df = df_standardized.copy()
newly_df = newly_df = newly_df.drop(newly_df.columns[-1], axis=1)
new_numerical_columns = newly_df.select_dtypes(include=['number']).columns
X_t = newly_df[new_numerical_columns].values
# Initialize PCA (keep n_components = len(numerical_columns) for full analysis)
pca = PCA(n_components=len(new_numerical_columns))
# Fit the PCA model to the data
pca.fit(X_t)
# Calculate explained variance (proportion of each component)
total_variance = np.sum(pca.explained_variance_)
explained_variance_ratio = (pca.explained_variance_ / total_variance) * 100  # Convert to percentage
# Calculate cumulative explained variance
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
# Find the number of components that explain at least 80% variance
threshold = 80
components_80_pct = np.argmax(cumulative_explained_variance_ratio >= threshold) + 1  # +1 for indexing
fig = go.Figure()
# Add the line plot with hover data
fig.add_trace(go.Scatter(
    x=list(range(1, 16)), 
    y=cumulative_explained_variance_ratio,
    mode='lines+markers',
    marker=dict(size=6),
    line=dict(dash='dash'),
    hovertemplate='<b>Component %{x}</b><br>Cumulative Explained Variance: %{y:.2f}%',
    name='Explained Variance'
))
# Add a point to highlight where the cumulative variance reaches 80%
fig.add_trace(go.Scatter(
    x=[components_80_pct],
    y=[cumulative_explained_variance_ratio[components_80_pct - 1]],  # -1 because of Python indexing
    mode='markers+text',
    text=[f'80% variance at {components_80_pct} PCs'],
    textposition='top right',
    marker=dict(size=12, color='red'),
    name='80% Variance Threshold'
))
# Update layout to make the plot more readable
fig.update_layout(
    title="Scree Plot with Cumulative Explained Variance (%)",
    xaxis_title="Number of Principal Components",
    yaxis_title="Cumulative Explained Variance (%)",
    template="plotly_white",
    hovermode="closest",
    showlegend=False,
    width=800,   
    height=600 
)
fig.show()

new_dimen = 10 # Number of dimensions chosen from the scree plot
pca_knn = PCA(n_components=new_dimen)
reduced_dimen = pca_knn.fit_transform(X_t)
# Ensure the indices of both DataFrames are the same
df_reduced = pd.DataFrame(reduced_dimen, columns=['pc1', 'pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10',])
# Reset index to align both DataFrames
df_reduced.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)
# Add the 'track_genre' column from the original DataFrame to the reduced DataFrame
df_reduced['genre'] = df_standardized['track_genre']
# Custom implementation of train_test_split
def custom_train_test_split(X, y, test_size=0.1, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    validate_indices = indices[test_set_size : 2 * test_set_size]
    train_indices = indices[2* test_set_size:]
    return X[train_indices], X[validate_indices], X[test_indices], y[train_indices], y[validate_indices], y[test_indices]
X_train,X_validate,X_test,y_train,y_validate,y_test = custom_train_test_split(df_reduced.iloc[:, 0:15].values,df_reduced.iloc[:,-1].values,test_size=0.1,random_state=2)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train1,X_validate1,X_test1,y_train1,y_validate1,y_test1 = custom_train_test_split(df_standardized.iloc[:, 0:15].values,df_standardized.iloc[:,-1].values,test_size=0.1,random_state=2)
X_train1 = X_train1.astype(np.float32)
X_test1 = X_test1.astype(np.float32)

import time
accuracy_results = []
distance_metrics = ['manhattan']
for distance_metric in distance_metrics:
    print(f"Distance Metric: {distance_metric}")
    start_time = time.time()
    apnaKnn = Knn(k=29, distance_metric=distance_metric)
    apnaKnn.fit(X_train, y_train)
    y_pred1 = apnaKnn.predict(X_validate)
    accuracy = np.mean(y_pred1 == y_validate)   # Calculate accuracy
    accuracy_results.append((accuracy, 29, distance_metric))
    print(f".k: {29}, accuracy: {accuracy} ")
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds") # Print execution time
# Measure the time for the complete dataset
start_time_complete = time.time()
knn_complete = Knn(k=29, distance_metric='manhattan')
knn_complete.fit(X_train1, y_train1)  # X_train1 is the complete dataset
y_pred_complete = knn_complete.predict(X_test1)  # X_test1 is the complete dataset
end_time_complete = time.time()
inference_time_complete = end_time_complete - start_time_complete
print(f"Inference time on complete dataset: {inference_time_complete:.4f} seconds")
# Measure the time for the reduced dataset
start_time_reduced = time.time()
knn_reduced = Knn(k=29, distance_metric='manhattan')
knn_reduced.fit(X_train, y_train)  # X_train reduced
y_pred_reduced = knn_reduced.predict(X_test)  # X_test reduced
end_time_reduced = time.time()
inference_time_reduced = end_time_reduced - start_time_reduced
print(f"Inference time on reduced dataset: {inference_time_reduced:.4f} seconds")
# Plot the inference times
inference_times = [inference_time_complete, inference_time_reduced]
datasets = ['Complete Dataset', 'PCA-Reduced Dataset']
plt.figure(figsize=(8, 5))
plt.bar(datasets, inference_times, color=['blue', 'green'])
plt.title('Inference Time Comparison: Complete vs PCA-Reduced Dataset')
plt.ylabel('Inference Time (seconds)')
plt.xlabel('Dataset Type')
plt.show()

# 9.2 Evaluation

# Initialize KNN with k=29 and Manhattan distance
apnaKnn = Knn(k=29, distance_metric='manhattan')
apnaKnn.fit(X_train, y_train)
y_pred_validate = apnaKnn.predict(X_validate)
# for reduced dataset
accuracy = accuracy_score(y_validate, y_pred_validate)
precision = precision_score(y_validate, y_pred_validate)
recall = recall_score(y_validate, y_pred_validate)
f1 = f1_score(y_validate, y_pred_validate)
print(f"FOR REDUCED DATASET:  \n")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
# for complete dataset
apnaKnn.fit(X_train1, y_train1)
y_pred_validate = apnaKnn.predict(X_validate1)
# Calculate evaluation metrics using custom functions
accuracy = accuracy_score(y_validate1, y_pred_validate)
precision = precision_score(y_validate1, y_pred_validate)
recall = recall_score(y_validate1, y_pred_validate)
f1 = f1_score(y_validate1, y_pred_validate)
print(f"FOR COMPLETE DATASET:  \n")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1 Score: {f1:.4f}")



