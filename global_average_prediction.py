import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# reading the data
temps_df = pd.read_csv('./datasets/GlobalLandTemperaturesByCountry.csv')

# dropping missed data to clean the dataset
temps_clean_df = temps_df.dropna()

temps_clean_df = temps_clean_df.copy()
temps_clean_df['dt'] = pd.to_datetime(temps_clean_df['dt'])
temps_clean_df['Year'] = temps_clean_df['dt'].dt.year

# getting the average year temperature
temps_clean_df['AverageTemperatureYear'] = temps_clean_df['Year']
temps_clean_df['AverageTemperatureYear'] = temps_clean_df['AverageTemperatureYear'].map(temps_clean_df.groupby(['Year']).AverageTemperature.mean())

years = [x for x in range(temps_clean_df['Year'].min(), temps_clean_df['Year'].max() + 1)]
average_years = pd.DataFrame({'Year': years})
average_years['AverageTemperatureYear'] = average_years['Year']
average_years['AverageTemperatureYear'] = average_years['AverageTemperatureYear'].map(temps_clean_df.groupby(['Year']).AverageTemperature.mean())

# Predicting temperatures in the future 
# splitting the dataframe
X, Y = temps_clean_df[['Year']], temps_clean_df['AverageTemperature']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

final_year = temps_clean_df['Year'].max()
years_ahead = [final_year + 50, final_year + 100, final_year + 150, final_year + 200]

future_years = pd.DataFrame({'Year': years_ahead})
predicted_temperatures = model.predict(future_years)

# visualising the data
plt.figure(figsize=(10, 6))
plt.scatter(temps_clean_df['Year'], temps_clean_df['AverageTemperature'], label='Observed Data')
plt.plot(average_years['Year'], average_years['AverageTemperatureYear'], 'r-', label='Average Year Temperatures')
plt.plot(future_years['Year'], predicted_temperatures, 'gx--', label='Predicted Temperatures')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.title('Simulation of global warming effects')
plt.legend()
plt.grid(True)
plt.show()