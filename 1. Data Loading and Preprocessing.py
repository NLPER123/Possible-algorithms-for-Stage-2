import numpy as np
import scipy.io as sio
from chronux import rmlinesc  # Chronux toolbox for line noise removal

# Load ECoG data
def load_ecog_data(file_path):
    data = sio.loadmat(file_path)
    return data['ecog_data']  # Assuming the ECoG data is stored under 'ecog_data'

# Preprocess data (subtract mean and remove line noise)
def preprocess_data(ecog_data, n_tapers=19, time_bandwidth=5):
    # Subtract mean from each epoch
    ecog_data = ecog_data - np.mean(ecog_data, axis=1, keepdims=True)
    
    # Remove line noise using rmlinesc from the Chronux toolbox
    ecog_data_clean = rmlinesc(ecog_data, n_tapers=n_tapers, time_bandwidth=time_bandwidth)
    return ecog_data_clean

