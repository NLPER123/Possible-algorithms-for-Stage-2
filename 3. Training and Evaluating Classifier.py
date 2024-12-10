from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors

# Train nearest-median classifier
def train_classifier(features, labels):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
    
    # Train nearest-median classifier using NearestNeighbors from sklearn
    model = NearestNeighbors(n_neighbors=1, metric='manhattan')  # Nearest neighbor classifier
    model.fit(X_train)  # Fit model on training data
    
    # Evaluate accuracy on test set
    test_predictions = model.kneighbors(X_test)[1]
    accuracy = np.mean(test_predictions == y_test)  # Compare predicted vs actual labels
    return accuracy

# Cross-validation to evaluate classifier performance
def evaluate_classifier(features, labels):
    scores = cross_val_score(NearestNeighbors(n_neighbors=1, metric='manhattan'), features, labels, cv=10)
    return np.mean(scores)  # Return average cross-validation score
