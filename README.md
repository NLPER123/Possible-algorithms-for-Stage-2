This code assumes some preprocessing and feature extraction steps, and includes a classification model

Summary of Steps:

1. Data Loading: Load the ECoG data files (both macaque and human) for analysis.

2. Preprocessing: Apply mean subtraction and line noise removal to the data.

3. Feature Extraction: Use the hctsa toolbox to extract relevant time-series features.

4. Training the Classifier: Train a nearest-median classifier and evaluate its performance using cross-validation.

5. Epoch Splitting: Split the data into epochs, train classifiers using varying numbers of epochs, and assess performance.
