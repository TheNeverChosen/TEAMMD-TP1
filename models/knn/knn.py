import numpy as np
from scipy.spatial.distance import cdist

class KNearestNeighours:
    def __init__(self, k) -> None:
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        
        self.unique_labels = np.unique(y)
        
        self.centroids = np.empty((len(self.unique_labels), self.X.shape[1]))
        
        for i, label in enumerate(self.unique_labels):
            label_mask = y == label
            self.centroids[i] = X[label_mask].mean(axis=0)
        
    def predict(self, X, metric='euclidean', k_means=False):
        
        if k_means:
            distances = cdist(X, self.centroids, metric=metric)
            sorting_indices = np.argsort(distances, axis=1)
            y_pred = np.take(self.unique_labels, sorting_indices[:, 0])
            return np.array(y_pred)
    
        
        # p-norm distance between each point in X and every point in self.X
        distances = cdist(X, self.X, metric=metric)

        # get k closest points indices
        sorting_indices = np.argsort(distances, axis=1)
        k_points = sorting_indices[:, :self.k]
        k_points_shape = k_points.shape
        
        # substitute point indices with its label
        k_labels = np.take(self.y, k_points.flatten())
        k_labels = np.array(k_labels).reshape(k_points_shape)
        
        # returns label with highest count
        def get_label(labels):
            label_values, counts = np.unique(labels, return_counts=True)
            prediction = label_values[np.argmax(counts)]
            return prediction

        y_pred = np.apply_along_axis(get_label, axis=1, arr=k_labels)
        return y_pred
            
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X, y = load_iris(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    knn = KNearestNeighours(1)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test, k_means=True)
    
    print(accuracy_score(y_test, y_pred))
