import numpy as np 
import pandas as pd 

class KMeans:
    
    def __init__(self, K=8, max_iter=300, number_of_fits=1, random_state=None):
        # Initializing the hyperparameters
        self.K = K
        self.max_iter=300
        self.random_state = random_state
        self.number_of_fits = number_of_fits
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # Normalize the data
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        X = (X - self.means) / self.stds 
        
        # Fitting the data self.number_of_fits-times, and returning the centroids and clusters from the fit that had the best silhouette-score.
        fits = [self.single_fit(X) for _ in range(self.number_of_fits)]
        index = np.argmax([euclidean_silhouette(X, fits[i][1]) for i in range(self.number_of_fits)])
        self.centroids, self.clusters = fits[index][0], fits[index][1]
    
    def single_fit(self, X):
        """
        Fitting the data, i.e. estimates the parameters for the classifier
        """
        centroids = self.init_centroids(X) # Choosing the initial centroids
        clusters = np.zeros(X.shape[0], dtype=int)

        for _ in range(self.max_iter):
            for i, row in X.iterrows():
                clusters[i] = self.find_nearest_cluster(row.values, centroids) # Finding the nearest cluster for each of the data-points.

            # Updating centroids to be the mean of the data-points in the current cluster
            new_centroids = np.array([
                X.iloc[np.where(clusters == j)[0]].mean().to_numpy()
                for j in range(self.K)
            ])

            # Check convergence, i.e. if the centroids and new_centroids are (almost) the same. If so, exit the loop.
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return [centroids, clusters]


    def find_nearest_cluster(self, values, centroids): 
        """
        Finding the centroid closest to the data-post, and assigning it to the centroids respecive cluster.
        """
        return np.argmin([euclidean_distance(values, centroids[j]) for j in range(self.K)])
    
    def init_centroids(self, X):
        """
        Initializing the centroids by using an algorithm inspired by kmeans++.
        First, randomly select a data-post where the fist centroid gets placed. Then, for the rest of the centroids, always place the centroid at the data-point
        that is the furthest away from the rest of the centroids.
        """
        centroids = [X.iloc[np.random.randint(len(X))]] # Create a list that will contain all the centroids, and insert a random data-point as the first centroid.

        # Initialize the list of squared distances for each data point to its closest centroid
        distances_to_closest_centroid = [euclidean_distance(x, centroids[0])**2 for x in X.values]

        # Loop to select the remaining centroids
        for i in range(1, self.K):
            # Calculate the cumulative probabilities for the next centroid selection
            cum_probabilities = np.cumsum(distances_to_closest_centroid) / np.sum(distances_to_closest_centroid)

            # Generate a random number
            rand_value = np.random.rand()

            # Find the index of the data point corresponding to the next centroid
            new_centroid_index = np.argmax(cum_probabilities >= rand_value)

            # Add the selected centroid to the centroids list
            centroids.append(list(X.iloc[new_centroid_index]))

            # Update the squared distances for the newly selected centroid
            new_distances = [euclidean_distance(x, centroids[-1])**2 for x in X.values]
            distances_to_closest_centroid = np.minimum(distances_to_closest_centroid, new_distances)
        
        return np.array(centroids)

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X = (X - self.means) / self.stds # Normalizing the data.

        # For each of the data-points, assign it to the cluster related to its nearest centroid.
        clusters = np.zeros(X.shape[0], dtype=int)
        for i, row in X.iterrows():
            clusters[i] = self.find_nearest_cluster(row.values, self.centroids)
        return clusters
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return np.array([(self.centroids[i] * self.stds) + self.means for i in range(len(self.centroids))]) # Returning array of de-normalized centroids.

# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    # for i, c in enumerate(clusters):
    #     Xc = X[z == c]
    #     mu = Xc.mean(axis=0)
    #     distortion += ((Xc - mu) ** 2).sum(axis=1)

    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        mu_arr = np.array([mu for _ in range(len(Xc))])
        distortion += np.sum(((Xc - mu_arr) ** 2))
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))