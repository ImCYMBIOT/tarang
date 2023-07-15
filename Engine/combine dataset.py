import pandas as pd

# List of dataset filenames
dataset_files = [r"D:\github\tarang\data2\data_by_artist.csv", r"D:\github\tarang\data2\data_by_genres.csv", r"D:\github\tarang\data2\data_by_year.csv",r"D:\github\tarang\data2\data_w_genres.csv"]

# List of common features
common_features = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
                   'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity', 'key']

# Create an empty unified dataset
unified_dataset = pd.DataFrame(columns=common_features)

# Iterate through dataset files and merge into unified dataset
for file in dataset_files:
    # Read the dataset file
    dataset = pd.read_csv(file)
    
    # Merge dataset with unified dataset based on common features
    unified_dataset = pd.merge(unified_dataset, dataset[common_features], how='outer')

# Handle missing values
unified_dataset.fillna(value=0, inplace=True)  # Fill missing values with 0 (you can change it as needed)

# Validate the unified dataset
print(unified_dataset.head())  # Check the first few rows of the unified dataset
print(unified_dataset.info())  # Display information about the unified dataset

# Save the unified dataset
unified_dataset.to_csv('unified_dataset.csv', index=False)  # Save as CSV file
