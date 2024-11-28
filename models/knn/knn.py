# import numpy as np

# class Knn:

#     def __init__(self, k = 3, distance_metric = 'euclidean' ):
#         self.n_neighbors = k
#         self.X_train = None
#         self.y_train = None
#         self.distance_metric = distance_metric

#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train

#     def predict(self, X_test):

        



#         y_pred = []

#         for i in X_test:
#             # Calculate distance with each training point
#             distances = [self.calculate_distance(i, j) for j in self.X_train]
#             n_neighbors = sorted(list(enumerate(distances)), key=lambda x: x[1])[:self.n_neighbors]
#             label = self.majority_count(n_neighbors)
#             y_pred.append(label)
        
#         return np.array(y_pred)

#     def calculate_distance(self, first, second):

#         if self.distance_metric == 'euclidean':
#             return np.linalg.norm(first - second)
#         elif self.distance_metric == 'manhattan':
#             return np.sum(np.abs(first - second))
#         elif self.distance_metric == 'cosine':
#             if np.linalg.norm(first) == 0 or np.linalg.norm(second) == 0:
#                 return 1.0
#             return 1 - np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        
            

#     def majority_count(self, neighbors):
#         votes = [self.y_train[i[0]] for i in neighbors]
#         counts = custom_counter(votes)
#         # Find the element with the maximum count
#         max_count = max(counts.values())
#         # Find the elements with the maximum count
#         max_elements = [key for key, value in counts.items() if value == max_count]
#         # If there's a tie, the function will return the first element from the max_elements list
#         return max_elements[0]

# def custom_counter(votes):
#     count_dict = {}
#     for vote in votes:
#         if vote in count_dict:
#             count_dict[vote] += 1
#         else:
#             count_dict[vote] = 1
#     return count_dict




import numpy as np

class Knn:

    def __init__(self, k=3, distance_metric='euclidean'):
        self.n_neighbors = k
        self.X_train = None
        self.y_train = None
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if self.distance_metric == 'euclidean':
            distances = self.euclidean_distance(X_test)
        elif self.distance_metric == 'manhattan':
            distances = self.manhattan_distance(X_test)
        elif self.distance_metric == 'cosine':
            distances = self.cosine_distance(X_test)

        
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors] # Get the indices of the k nearest neighbors
        neighbors_labels = self.y_train[neighbors_indices] # Gather the nearest neighbors' labels
        y_pred = np.array([self.majority_vote(labels) for labels in neighbors_labels])   # Get the most common label (majority vote) for each test instance

        return y_pred



  # I have taken the code for vectorised calculation of metrices from google 
  # My initial basic code is in comments
    def euclidean_distance(self, X_test, batch_size=500):
        num_test_samples = X_test.shape[0]
        num_train_samples = self.X_train.shape[0]
        dists = np.zeros((num_test_samples, num_train_samples))
        
        # Process in batches
        for start_idx in range(0, num_test_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_test_samples)
            X_test_batch = X_test[start_idx:end_idx]
            
            # Compute squared distances for the current batch
            dists_batch = -2 * np.dot(X_test_batch, self.X_train.T) + np.sum(self.X_train * 2, axis=1) + np.sum(X_test_batch * 2, axis=1)[:, np.newaxis]
            
            # Ensure no negative distances are used (due to floating point precision errors)
            dists_batch = np.maximum(dists_batch, 0)
            
            # Store the results in the main distance matrix
            dists[start_idx:end_idx] = np.sqrt(dists_batch)
        
        return dists

    def manhattan_distance(self, X_test, batch_size=500):
        # X_test = X_test.astype(np.float64)
        num_test_samples = X_test.shape[0]
        num_train_samples = self.X_train.shape[0]
        dists = np.zeros((num_test_samples, num_train_samples))
        
        # Process in batches
        for start_idx in range(0, num_test_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_test_samples)
            X_test_batch = X_test[start_idx:end_idx]
            
            # Compute distances for the current batch
            dists_batch = np.sum(np.abs(X_test_batch[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
            dists[start_idx:end_idx] = dists_batch
        
        return dists


    def cosine_distance(self, X_test):
        X_test_norm = np.linalg.norm(X_test, axis=1, keepdims=True)
        X_train_norm = np.linalg.norm(self.X_train, axis=1, keepdims=True)
        cosine_sim = np.dot(X_test, self.X_train.T) / (X_test_norm * X_train_norm.T)
        cosine_dist = 1 - cosine_sim
        return cosine_dist

    def majority_vote(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count_index = np.argmax(counts)
        return unique_labels[max_count_index]





