import pandas as pd
import numpy as np

df = pd.read_csv('automobile/auto.csv')
#print("The first 5 rows of the dataframe")
#print(df.head(5))

#print("The Last 5 rows of the dataframe")
#print(df.tail(5))


#This information is available at: https://archive.ics.uci.edu/ml/datasets/Automobile.
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
#print("headers\n", headers)

df.columns = headers
#print("The first 5 rows of the dataframe after setting the headers")
#print(df.head(5))

df.replace('?', np.nan, inplace=True)
#df=df1.dropna(subset=["price"], axis=0)
print(df.head(5))

missing_data = df.isnull()
missing_data.head(5)

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")

# Calculate the mean value for the "normalized-losses" column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

#Calculate the mean value for the "bore" column
avg_bore = df["bore"].astype("float").mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)

#calculate the mean value for the "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

#calculate the mean value for the "horsepower" column
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
print("Average of horsepower:", avg_horsepower)
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)

#calculate the mean value for the "peak-rpm" column
avg_peakrpm = df["peak-rpm"].astype("float").mean(axis=0)
print("Average of peak-rpm:", avg_peakrpm)
df["peak-rpm"].replace(np.nan, avg_peakrpm, inplace=True)

#calculate the mean value for the "num-of-door" column
mode_num_of_doors = df["num-of-doors"].value_counts().idxmax()
print("Average of num-of-doors:", mode_num_of_doors)
df["num-of-doors"].replace(np.nan, mode_num_of_doors, inplace=True)

#Finally, drop all rows that do not have price data
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
#print(df.head(5))

#print("Dtypes of the dataframe:", df.dtypes)
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
#print("Dtypes of the dataframe:", df.dtypes)

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
#print(df.head())

# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
#print(df.head())

# replace (original value) by (original value)/(maximum value)
df['height'] = df['height']/df['height'].max()
print(df['height'])

#Binning the data
df["horsepower"]=df["horsepower"].astype(int, copy=True)    
print(df["horsepower"])

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head(20))

# number of vehicles in each bin
print(df["horsepower-binned"].value_counts())

# plot in histogram
import matplotlib.pyplot as plt
from matplotlib import pyplot

plt.hist(df["horsepower"])
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#plt.show()

# Creating dummy variables for categorical variables "fuel-type"
df.columns
dummy_variable = pd.get_dummies(df["fuel-type"])
print(dummy_variable.head())

dummy_variable.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable.head()

# merge data frame "df" and "dummy_variable" 
df = pd.concat([df, dummy_variable], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

print(df.head(10))

# Creating dummy variables for categorical variables "aspiration"
dummy_variable_aspiration = pd.get_dummies(df["aspiration"])
print(dummy_variable_aspiration.head())
dummy_variable_aspiration.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace=True)
dummy_variable_aspiration.head()
# merge data frame "df" and "dummy_variable_aspiration"
df = pd.concat([df, dummy_variable_aspiration], axis=1)
# drop original column "aspiration" from "df"
df.drop("aspiration", axis = 1, inplace=True)
print(df.head(10))

df.to_csv('automobile/clean_df.csv', index=False)