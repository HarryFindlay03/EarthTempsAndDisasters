import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

temps_df = pd.read_csv('./datasets/GlobalLandTemperaturesByCountry.csv')

print(temps_df.shape)

print(temps_df.head())

# describing the data
print(temps_df.AverageTemperature.describe())

# checking for null values
print(temps_df[pd.isnull(temps_df.AverageTemperature)])

# dropping missed data to clean the dataset
temps_clean_df = temps_df.dropna()
# print(temps_clean_df[pd.isnull(temps_clean_df.AverageTemperature)])

# Plotting the uk data with Seaborn
uk_temps_df = temps_clean_df.loc[(temps_clean_df['Country'] == 'United Kingdom (Europe)')]
uk_temps_df = uk_temps_df.copy()
uk_temps_df['dt'] = pd.to_datetime(uk_temps_df['dt'])
uk_temps_df['Year'] = uk_temps_df['dt'].dt.year.astype(int)
uk_temps_df['AverageYearTemperature'] = uk_temps_df['Year']

uk_temps_df['AverageYearTemperature'] = uk_temps_df['AverageYearTemperature'].map(uk_temps_df.groupby(['Year']).AverageTemperature.mean())


plt.figure(figsize=(10, 6))
sns.lineplot(data=(uk_temps_df['AverageTemperature'], uk_temps_df['AverageYearTemperature']))
plt.show()
