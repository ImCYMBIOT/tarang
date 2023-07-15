import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Read data from CSV file
data = []
with open(r"D:\github\tarang\unified_dataset.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

# Find the indices of the desired columns
header = data[0]
popularity_index = header.index("popularity")
danceability_index = header.index("danceability")

# Separate data into features and labels
features = []
labels = []
for row in data[1:11]:
    features.append([float(row[popularity_index]), float(row[danceability_index])])
    labels.append(float(row[-1]))

# Scale features using Min-Max scaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Convert features to a numpy array
features_array = np.array(features)

# Create a line plot
plt.imshow(features_array, cmap='hot')
plt.colorbar(label='Values')
plt.xlabel('Popularity')
plt.ylabel('Danceability')
plt.title('Heatmap')

# Show the plot
plt.show()
