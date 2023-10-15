import numpy as np
from scipy.spatial.distance import cdist

class KNearestNeighours:
    def __init__(self, k) -> None:
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X):
        distances = cdist(X, self.X)

        sorting_indices = np.argsort(distances, axis=1)
        k_points = sorting_indices[:, :self.k]
        k_points_shape = k_points.shape
        k_labels = np.take(self.y, k_points.flatten())
        k_labels = np.array(k_labels).reshape(k_points_shape)
        
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
    
    y_pred = knn.predict(X_test)
    
    print(accuracy_score(y_test, y_pred))
