import pandas as pd
import matplotlib.pyplot as plt

from helper_functions import distance_between_points

# reading the data
disasters_df = pd.read_csv('./datasets/public_emdat_disasters.csv')

print(disasters_df.shape)

disasters_df = disasters_df.copy()
disasters_df['Year'] = disasters_df['DisNo.']
disasters_df['Year'] = disasters_df['Year'].apply(lambda x: int((str(x).split("-"))[0]))

years = [x for x in range(disasters_df['Year'].min(), disasters_df['Year'].max() + 1)]

for col in disasters_df.columns:
    print(col)

average_deaths_df = pd.DataFrame({'Year': years})
average_deaths_df['CumDeaths'] = average_deaths_df['Year']
average_deaths_df['CumDeaths'] = average_deaths_df['CumDeaths'].map(disasters_df.groupby(['Year']).Total_Deaths.sum())

# testing helper functions
# point_a = (39.09, 42.3)
# point_b = (17.5, -102.5)
# print(distance_between_points(point_a, point_b))

plt.figure(figsize=(12, 7))
plt.plot(average_deaths_df['Year'], average_deaths_df['CumDeaths'], 'b-', label='Total deaths')
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.title('Average Deaths per Year from Disasters')
plt.legend()
plt.grid(True)
plt.show()