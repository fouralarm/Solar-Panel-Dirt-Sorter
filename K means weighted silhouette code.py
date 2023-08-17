
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KMeans():
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []
        
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape       
    
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]       
        
        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
                
            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
    
            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break            
            if self.plot_steps:
                self.plot()        
                
        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels    
    
    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters   
    
    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index   
    
    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids    
    
    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0   
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))        
        
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)      
            
        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)        
        plt.show()
    def cent(self):
        return self.centroids     


from sklearn.metrics import pairwise_distances

def weighted_silhouette_score(X, labels, sigma=1.0):
    n_samples = len(X)
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k == 1:
        return 0.0
    #distances is the pairwise distance matrix
    distances = pairwise_distances(X)

    a = np.zeros(n_samples)
    b = np.zeros(n_samples)

    
    for i in range(n_samples):
        mask = (labels == labels[i])
        a_i = np.mean(distances[i, mask]) 
        a[i] = a_i
        
        #mask is a boolean array indicating which samples have the same label as sample i
        #np.mean is the function that computes the average distance
        
        b_i = np.min([np.mean(distances[i, labels == j]) for j in unique_labels if j != labels[i]])
        b[i] = b_i

    s = (b - a) / np.maximum(a, b)
    
    #this is our weighted array
    w = np.zeros(n_samples)
    for i in range(n_samples):
        w_i = np.exp(-distances[i]**2 / (2*sigma**2))
        w[i] = np.sum(w_i[labels == labels[i]]) - w_i[i]
    
    sw = s * w

    score = np.mean(sw)

    return score



import cv2
import multiprocessing as mp


# Define a worker function for each process
def worker(k, pixel_values,image):
    kmeans = KMeans(K=k, max_iters=100)
    
    #fitting the KMeans model to the pixel values
    y_pred = kmeans.predict(pixel_values)
    
    centers = np.uint8(kmeans.cent())
    
    y_pred=y_pred.astype(int)
    labels = y_pred.flatten()
    
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    return labels

if __name__ == '__main__':
    image = cv2.imread("leaf1.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    k_values = [2, 3, 4, 5]
    
    pool = mp.Pool(processes=4)
    
    labelarray = [pool.apply_async(worker, args=(kv, pixel_values,image)) for kv in k_values]
    
    labelfinal=[lab.get() for lab in labelarray]
    print(labelfinal)
    
    pool2=mp.Pool(processes=4)
    sscores=[pool2.apply_async(weighted_silhouette_score,args=(pixel_values,l)) for l in labelfinal]
    scores=[score.get() for score in sscores]
    print(scores)
    
    max_index=scores.index(max(scores))
    optimal=max_index+2
    print("optimal value of k is: ",optimal)
    
    kmeans = KMeans(K=optimal, max_iters=100)
    
    y_pred = kmeans.predict(pixel_values)
    
    centers = np.uint8(kmeans.cent())
    
    y_pred=y_pred.astype(int)
    labs = y_pred.flatten()
    
    segmented_image = centers[labs.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    plt.subplot(1,2,1)
    plt.imshow(image)    
    plt.subplot(1,2,2)
    plt.imshow(segmented_image)

    
    
    

