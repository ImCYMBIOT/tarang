import csv
import matplotlib.pyplot as plt

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

# Separate data into features
danceability = []
popularity = []
for row in data[1:]:
    danceability.append(float(row[danceability_index]))
    popularity.append(float(row[popularity_index]))

# Create a scatter plot
plt.scatter(danceability, popularity)
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.title('Relation between Danceability and Popularity')

# Show the plot
plt.show()
