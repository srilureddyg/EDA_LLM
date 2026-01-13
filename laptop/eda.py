import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import scipy.stats as stats

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
#plt.hist(df['Price'])
#plt.xlabel('Price')
#plt.ylabel('Count')
#plt.title('Price Bins')
#plt.show()

# Creating dummy variables for categorical variable 'Screen'
dummy_variable = pd.get_dummies(df, columns=['Screen'])
dummy_variable.rename(columns=lambda x: 'Screen_' + str(x) if x in df['Screen'].unique() else x, inplace=True)
df = pd.concat([df, dummy_variable], axis=1)
df.drop('Screen', axis=1, inplace=True)
print(dummy_variable.head(10))

df.to_csv('laptop/laptop_cleaned.csv', index=False)

df_clean = pd.read_csv('laptop/laptop_cleaned.csv')
print(df_clean.head())
print(df_clean.dtypes)

df_clean[['CPU_frequency', 'screen_size_inches', 'Weight_lb','Price']].corr()
print(df_clean[['CPU_frequency', 'screen_size_inches', 'Weight_lb','Price']].corr())
#sns.regplot(x='CPU_frequency', y='Price', data=df_clean)
#sns.regplot(x='screen_size_inches', y='Price', data=df_clean)
#sns.regplot(x='Weight_lb', y='Price', data=df_clean)
plt.show()
'''Interpretation: "CPU_frequency" has a 36% positive correlation with the price of the laptops. The other two parameters have weak correlation with price.'''

# box plot "Category", "GPU", "OS", "CPU_core", "RAM_GB", "Storage_GB_SSD"
'''plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
sns.boxplot(x='Category', y='Price', data=df_clean)
plt.subplot(2,3,2)
sns.boxplot(x='GPU', y='Price', data=df_clean)
plt.subplot(2,3,3)
sns.boxplot(x='OS', y='Price', data=df_clean)
plt.subplot(2,3,4)
sns.boxplot(x='CPU_core', y='Price', data=df_clean)
plt.subplot(2,3,5)
sns.boxplot(x='RAM_GB', y='Price', data=df_clean)
plt.subplot(2,3,6)
sns.boxplot(x='Storage_GB_SSD', y='Price', data=df_clean)
plt.tight_layout()'''
#plt.show()

print(df_clean.describe())
print(df_clean.describe(include=['object']))

df_clean[['GPU', 'CPU_core','Price']]
df_gptest = df_clean[['GPU','CPU_core','Price']]
grouped_test1 = df_gptest.groupby(['GPU','CPU_core'],as_index=False).mean()
print(grouped_test1)

# pivot table
pivot_table = pd.pivot(data=grouped_test1, index='GPU', columns='CPU_core', values='Price')

#plot
#sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")
#plt.title('Average Price by GPU and CPU Core Count')
#plt.xlabel('CPU Core Count')
#plt.ylabel('GPU')
#plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_test1, cmap='RdBu')

#label names
row_labels = grouped_test1.columns
col_labels = grouped_test1.index    

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_test1.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_test1.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

fig.colorbar(im)
#plt.show()

# Pearson correlation test and p-value for 'RAM_GB','CPU_frequency','Storage_GB_SSD','Screen_Size_inch','Weight_pounds','CPU_core','OS','GPU','Category'
pearson_cols = ['RAM_GB','CPU_frequency','Storage_GB_SSD','screen_size_inches','Weight_lb','CPU_core']
for col in pearson_cols:
    corr, p_value = stats.pearsonr(df_clean[col], df_clean['Price'])
    print(f'Pearson correlation between {col} and Price: {corr:.4f}, p-value: {p_value:.4f}')   

# Conclusion: "CPU_frequency" has a significant positive correlation with the price of laptops (corr=0.3612, p-value=0.0000). 
# Other features show weak or no significant correlation with price.
