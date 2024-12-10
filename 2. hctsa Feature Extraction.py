import hctsa  # Assuming you have the hctsa toolbox available

# Extract hctsa features from the cleaned ECoG data
def extract_hctsa_features(ecog_data_clean):
    features = hctsa.compute_features(ecog_data_clean)
    return features
