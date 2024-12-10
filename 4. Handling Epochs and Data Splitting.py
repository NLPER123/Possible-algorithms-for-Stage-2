import random

# Split the data into epochs and train the classifier on different numbers of epochs
def split_data_into_epochs(ecog_data_clean, num_epochs=512):
    # Assuming data is structured with rows as epochs and columns as time points
    epochs = np.array_split(ecog_data_clean, len(ecog_data_clean) // num_epochs, axis=0)
    return epochs

# Train classifier on different numbers of epochs
def classifier_performance_by_epoch_count(epochs, classifier, num_epochs_list=[10, 50, 100, 512]):
    performance = {}
    
    for num_epochs in num_epochs_list:
        # Select a random sample of epochs
        selected_epochs = random.sample(epochs, num_epochs)
        
        # Extract features for these epochs
        features = extract_hctsa_features(np.vstack(selected_epochs))
        
        # Train classifier
        accuracy = train_classifier(features, labels)
        performance[num_epochs] = accuracy
    
    return performance
