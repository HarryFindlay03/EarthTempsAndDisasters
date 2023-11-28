import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from helper_functions import distance_between_points

# reading the data
disasters_df = pd.read_csv('./datasets/public_emdat_disasters.csv')

print(disasters_df.shape)

disasters_df = disasters_df.copy()
disasters_df['Year'] = disasters_df['DisNo.']
disasters_df['Year'] = disasters_df['Year'].apply(lambda x: int((str(x).split("-"))[0]))

# drop rows with nan lat long coords
disasters_df.dropna(how='any', subset=['Latitude', 'Longitude'], inplace=True)

points = disasters_df[['Latitude', 'Longitude']].to_numpy()

clustering = DBSCAN(eps=3, min_samples=2, metric=distance_between_points).fit(points)
num_clusters = len(set(clustering.labels_))
clusters = pd.Series(points[clustering.labels_ == n] for n in range(num_clusters))

print(clusters.head())

print(f'Number of Clusters: {num_clusters}')


# sum of deaths per year
years = [x for x in range(disasters_df['Year'].min(), disasters_df['Year'].max() + 1)]

average_deaths_df = pd.DataFrame({'Year': years})
average_deaths_df['CumDeaths'] = average_deaths_df['Year']
average_deaths_df['CumDeaths'] = average_deaths_df['CumDeaths'].map(disasters_df.groupby(['Year']).Total_Deaths.sum())

# testing helper functions
# point_a = (39.09, 42.3)
# point_b = (17.5, -102.5)
# print(distance_between_points(point_a, point_b))

# plt.figure(figsize=(12, 7))
# plt.plot(average_deaths_df['Year'], average_deaths_df['CumDeaths'], 'b-', label='Total deaths')
# plt.xlabel('Year')
# plt.ylabel('Deaths')
# plt.title('Average Deaths per Year from Disasters')
# plt.legend()
# plt.grid(True)

# plt.figure(figsize=(10, 6))
# plt.scatter(disasters_df['Longitude'], disasters_df['Latitude'], label="Locations of disasters")
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Lat Long coordiantes of Disasters from 1900')
# plt.legend()
# plt.grid(True)

# visualising clusters
color_list = np.array(['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple'])

plt.show()