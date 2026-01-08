import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot

df = pd.read_csv('laptop/laptop_data.csv')
print(df.info())
print(df.head())

df[['Screen_Size_cm']] = np.round(df[['Screen_Size_cm']], 2)
df[['Weight_kg']] = np.round(df[['Weight_kg']], 2)

df[['Weight_kg']] = df[['Weight_kg']].fillna(df[['Weight_kg']].mean())
most_frequent = df['Screen_Size_cm'].mode()[0]
df['Screen_Size_cm'] = df['Screen_Size_cm'].fillna(most_frequent)
#print(df.isna().sum())

print(df.dtypes)

df['screen_size_inches'] = df['Screen_Size_cm'] / 2.54
df['screen_size_inches'] = np.round(df['screen_size_inches'], 2)
df.drop(columns=['Screen_Size_cm'], inplace=True)
#print(df.head())

df[['Weight_lb']] = df[['Weight_kg']] * 2.205
df[['Weight_lb']] = np.round(df[['Weight_lb']], 2)
df.drop(columns=['Weight_kg'], inplace=True)
#print(df.head())

# Normalize the 'CPU_frequency' column using Min-Max scaling
df['CPU_frequency'] = df['CPU_frequency'] / df['CPU_frequency'].max()
print(df.head())

# Binning the 'Price' column into categories
bins = np.linspace(min(df['Price']), max(df['Price']), 4)
group_names = ['Low', 'Medium', 'High']
df['Price_binned'] = pd.cut(df['Price'], bins, labels=group_names, include_lowest=True)
#print(df[['Price', 'Price_binned']].head(20))   
print(df['Price_binned'].value_counts())

# Plotting histogram of 'Price'
plt.hist(df['Price'])
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Price Bins')
#plt.show()

# Creating dummy variables for categorical variable 'Screen'
dummy_variable = pd.get_dummies(df, columns=['Screen'])
dummy_variable.rename(columns=lambda x: 'Screen_' + str(x) if x in df['Screen'].unique() else x, inplace=True)
df = pd.concat([df, dummy_variable], axis=1)
df.drop('Screen', axis=1, inplace=True)
print(dummy_variable.head(10))
