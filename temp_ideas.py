import pandas as pd
from scipy.spatial.distance import euclidean
from itertools import combinations

# Example triplets
triplets = [(1, 2, 3), (4, 5, 6), (7, 8, 9), ...]  # List of 1000 triplets

# Compute distances between all pairs of triplets
distances = []
for pair in combinations(triplets, 2):
    distance = euclidean(pair[0], pair[1])
    distances.append(distance)

# Create a dataframe with triplets and distances
df = pd.DataFrame(triplets, columns=['Column1', 'Column2', 'Column3'])
df['Distance'] = distances

# Sort the dataframe by distance in descending order
df_sorted = df.sort_values(by='Distance', ascending=False)

# Select the top 10 most different triplets
most_different_triplets = df_sorted.head(10)

# Display the selected triplets
print(most_different_triplets)
