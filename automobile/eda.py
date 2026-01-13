import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from scipy import stats

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
#print(df.head(5))

missing_data = df.isnull()
missing_data.head(5)

for column in missing_data.columns.values.tolist():
    #print(column)
    print (missing_data[column].value_counts())
    #print("")

# Calculate the mean value for the "normalized-losses" column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
#print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

#Calculate the mean value for the "bore" column
avg_bore = df["bore"].astype("float").mean(axis=0)
#print("Average of bore:", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)

#calculate the mean value for the "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis=0)
#print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

#calculate the mean value for the "horsepower" column
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
#print("Average of horsepower:", avg_horsepower)
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)

#calculate the mean value for the "peak-rpm" column
avg_peakrpm = df["peak-rpm"].astype("float").mean(axis=0)
#print("Average of peak-rpm:", avg_peakrpm)
df["peak-rpm"].replace(np.nan, avg_peakrpm, inplace=True)

#calculate the mean value for the "num-of-door" column
mode_num_of_doors = df["num-of-doors"].value_counts().idxmax()
#print("Average of num-of-doors:", mode_num_of_doors)
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
#print(df['height'])

#Binning the data
df["horsepower"]=df["horsepower"].astype(int, copy=True)    
#print(df["horsepower"])

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
#print(bins)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
#print(df[['horsepower','horsepower-binned']].head(20))

# number of vehicles in each bin
#print(df["horsepower-binned"].value_counts())

# plot in histogram

#plt.hist(df["horsepower"])
#plt.xlabel("horsepower")
#plt.ylabel("count")
#plt.title("horsepower bins")
#plt.show()

# Creating dummy variables for categorical variables "fuel-type"
df.columns
dummy_variable = pd.get_dummies(df["fuel-type"])
#print(dummy_variable.head())

dummy_variable.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable.head()

# merge data frame "df" and "dummy_variable" 
df = pd.concat([df, dummy_variable], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

#print(df.head(10))

# Creating dummy variables for categorical variables "aspiration"
dummy_variable_aspiration = pd.get_dummies(df["aspiration"])
#print(dummy_variable_aspiration.head())
dummy_variable_aspiration.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace=True)
dummy_variable_aspiration.head()
# merge data frame "df" and "dummy_variable_aspiration"
df = pd.concat([df, dummy_variable_aspiration], axis=1)
# drop original column "aspiration" from "df"
df.drop("aspiration", axis = 1, inplace=True)
#print(df.head(10))

df.to_csv('automobile/clean_df.csv', index=False)

df_clean = pd.read_csv('automobile/clean_df.csv')
#print(df_clean['peak-rpm'].dtypes)

numeric_df = df_clean.select_dtypes(include=['float64','int64'])
#print(numeric_df.corr())

# correlation between columns: bore, stroke, compression-ratio and horsepower.
numeric_df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
#plt.show()

# Positive Linear Relationship
#sns.regplot(x="engine-size", y="price", data=df_clean)
#plt.show()
#print(df_clean[["engine-size", "price"]].corr())

#sns.regplot(x="highway-mpg", y="price", data=df_clean)
#plt.show()
#print(df_clean[["highway-mpg", "price"]].corr())

# weak linear replationship
#sns.regplot(x="peak-rpm", y="price", data=df_clean)
#plt.show()
#print(df_clean[["peak-rpm", "price"]].corr())

# correlation between stroke and price
#sns.regplot(x="stroke", y="price", data=df_clean)
#plt.show()
#print(df_clean[["stroke", "price"]].corr())

# correlation between price and stroke using regplot
#sns.regplot(x="stroke", y="price", data=df_clean)
#plt.show()

# Relationship between body-style and price
#sns.boxplot(x="body-style", y="price", data=df_clean)
#plt.show()

#Relationship between engine-location and price
#sns.boxplot(x="engine-location", y="price", data=df_clean)
#plt.show()

# Relationship between drive-wheels and price
#sns.boxplot(x="drive-wheels", y="price", data=df_clean)
#plt.show()

#print(df_clean.describe(include='object'))

df_clean['drive-wheels'].value_counts()
drive_wheels_counts = df_clean['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.reset_index(inplace=True)
drive_wheels_counts.rename(columns={'index':'drive-wheels', 'drive-wheels':'value_counts'}, inplace=True)
#print(drive_wheels_counts) 
drive_wheels_counts.index.name = 'drive-wheels'
#print(drive_wheels_counts)

#engine-location as variable
engine_loc_counts = df_clean['engine-location'].value_counts().to_frame()
engine_loc_counts.reset_index(inplace=True)
engine_loc_counts.rename(columns={'index':'engine-location', 'engine-location':'value_counts'}, inplace =True)
#print(engine_loc_counts)   
engine_loc_counts.index.name = 'engine-location'
#print(engine_loc_counts.head(10))

#print(df_clean['drive-wheels'].unique())    

df_group_one = df_clean[['drive-wheels','body-style','price']]
#print(df_group_one)    

df_grouped = df_group_one.groupby(['drive-wheels'], as_index=False).agg({'price':'mean'})
#print(df_grouped)

df_gptest = df_clean[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'], as_index=False).mean()
#print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
#print(grouped_pivot)
grouped_pivot = grouped_pivot.fillna(0) # fill missing values with 0
#print(grouped_pivot)

grouped_body_style = df_group_one.groupby(['body-style'], as_index=False).agg({'price':'mean'})
#print(grouped_body_style)  

# heatmap
#plt.pcolor(grouped_pivot, cmap='RdBu')
#plt.colorbar()
#plt.show()

#fig, ax = plt.subplots()
#im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
#row_labels = grouped_pivot.columns.levels[1]
#col_labels = grouped_pivot.index

#move ticks and labels to the center
#ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
#ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
#ax.set_xticklabels(row_labels, minor=False)
#ax.set_yticklabels(col_labels, minor=False)

#rotate the label if too long
#plt.xticks(rotation=90)

#fig.colorbar(im)
#plt.show()

df_clean.select_dtypes(include=['number']).corr()

pearson_coef, p_value = stats.pearsonr(df_clean['wheel-base'], df_clean['price'])
#print("The Pearson Correlation Coefficient between wheel-base and price is", pearson_coef, " with a P-value of P =", p_value)
#Conclusion:
#Since the p-value is < 0.001, the correlation between wheel-base and price is statistically significant.

#pearson correlation between horsepower and price
pearson_coef, p_value = stats.pearsonr(df_clean['horsepower'], df_clean['price'])
#print("The Pearson Correlation Coefficient between horsepower and price is", pearson_coef, " with a P-value of P =", p_value)   
#conclusion:
#Since the p-value is < 0.001, the correlation between horsepower and price is statistically significant.

#pearson correlation between length and price
pearson_coef, p_value = stats.pearsonr(df_clean['length'], df_clean['price'])
#print("The Pearson Correlation Coefficient between length and price is", pearson_coef, " with a P-value of P =", p_value)   
#Conclusion:
#Since the p-value is < 0.001, the correlation between length and price is statistically significant.

#pearson correlation between width and price
pearson_coef, p_value = stats.pearsonr(df_clean['width'], df_clean['price'])
#print("The Pearson Correlation Coefficient between width and price is", pearson_coef, " with a P-value of P =", p_value)   
#Conclusion:
#Since the p-value is < 0.001, the correlation between width and price is statistically significant.

#pearson correlation between curb-weight and price
pearson_coef, p_value = stats.pearsonr(df_clean['curb-weight'], df_clean['price'])
#print("The Pearson Correlation Coefficient between curb-weight and price is", pearson_coef, " with a P-value of P =", p_value)   
#Conclusion:
#Since the p-value is < 0.001, the correlation between curb-weight and price is statistically significant.  

#pearson correlation between engine-size and price
pearson_coef, p_value = stats.pearsonr(df_clean['engine-size'], df_clean['price'])
#print("The Pearson Correlation Coefficient between engine-size and price is", pearson_coef, " with a P-value of P =", p_value)   
#Conclusion:
#Since the p-value is < 0.001, the correlation between engine-size and price is statistically significant.

#pearson correlation between bore and price
pearson_coef, p_value = stats.pearsonr(df_clean['bore'], df_clean['price'])
#print("The Pearson Correlation Coefficient between bore and price is", pearson_coef, " with a P-value of P =", p_value)   
#Conclusion:
#Since the p-value is < 0.001, the correlation between bore and price is statistically significant.

#pearson correlation between city-mpg and price
pearson_coef, p_value = stats.pearsonr(df_clean['city-mpg'], df_clean['price'])
#print("The Pearson Correlation Coefficient between city-mpg and price is", pearson_coef, " with a P-value of P =", p_value)   
#Conclusion:
#Since the p-value is < 0.001, the correlation between city-mpg and price is statistically significant.

#pearson correlation between highway-mpg and price
pearson_coef, p_value = stats.pearsonr(df_clean['highway-mpg'], df_clean['price'])
#print("The Pearson Correlation Coefficient between highway-mpg and price is", pearson_coef, " with a P-value of P =", p_value)   
#Conclusion:
#Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant.  

# Conclusion:Important Variables
"""
We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. 
We have narrowed it down to the following variables:

Continuous numerical variables:
    Length
    Width
    Curb-weight
    Engine-size
    Horsepower
    City-mpg
    Highway-mpg
    Wheel-base
    Bore
Categorical variables:
    Drive-wheels
As we now move into building machine learning models to automate our analysis, feeding the model with variables that 
meaningfully affect our target variable will improve our model's prediction performance.

"""

""" Model Development"""
usedcars_df = pd.read_csv('automobile/automobileEDA.csv')

# linear regression model
from sklearn.linear_model import LinearRegression

# Create the linear regression model object:
lm=LinearRegression()
lm

X = usedcars_df[['highway-mpg']]
Y = usedcars_df['price']
lm.fit(X,Y)

Yhat=lm.predict(X)
#print(Yhat[0:5]) # display the first 5 predicted values
#print(lm.intercept_)
#print(lm.coef_)
# y = 38423.31 - 821.73 x

lm1=LinearRegression()
X1 = usedcars_df[['engine-size']]
Y1 = usedcars_df['price']
lm1.fit(X1,Y1)

Yhat1=lm1.predict(X1)
#print(Yhat1[0:5]) # display the first 5 predicted values
#print(lm1.intercept_)
#print(lm1.coef_)
# y1 = -7963.34 + 166.86 x1
# price = -7963.34 + 166.86 * engine-size

# Multiple Linear Regression
Z = usedcars_df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, usedcars_df['price'])
#print(lm.intercept_)
#print(lm.coef_)
# price = -15806.62 + 53.53 * horsepower + 4.71 * curb-weight + 81.18 * engine-size + 36.06 * highway-mpg

lm2=LinearRegression()
Z1 = usedcars_df[['normalized-losses', 'highway-mpg']]
lm2.fit(Z1, usedcars_df['price'])   

#print(lm2.intercept_)
#print(lm2.coef_)
# price = 38201.31 + 53.5 * normalized-losses - 821.73 * highway-mpg

# Model evaluation using visualization
import seaborn as sns

width = 12
height = 10
#plt.figure(figsize=(width, height))
#sns.regplot(x="highway-mpg", y="price", data=usedcars_df)
#plt.ylim(0,)
#plt.show()

#sns.regplot(x="peak-rpm", y="price", data=usedcars_df)
#plt.ylim(0,)
#plt.show()

highway_corr = usedcars_df[["peak-rpm","highway-mpg", "price"]].corr()
print("This is correlation between highway-mpg and price" + str(highway_corr))

width = 12
height = 10
#plt.figure(figsize=(width, height))
#sns.residplot(x=usedcars_df['highway-mpg'], y=usedcars_df['price'])
#plt.show()
# We can see that the residuals are not randomly distributed. 
# This is an indication that there is a non-linear relationship between highway-mpg and price.

# multiple linear regression
Yhat = lm.predict(Z)
plt.figure(figsize=(width, height))
ax1 = sns.displot(usedcars_df['price'], color="r", label="Actual Value")
sns.displot(Yhat, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
#plt.show()
#plt.close()
# conclusion: The red line represents the actual price distribution, while the blue line represents the predicted price distribution.
# The closer the blue line is to the red line, the better our model is performing.  
# In this case, the two lines are very close to each other, which means our model is performing very well.

#Polynomial Regression and Pipelines

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    #plt.show()
    #plt.close()

x = usedcars_df['highway-mpg']
y = usedcars_df['price']
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
#print(p)
#PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)
# conclusion:We can conclude that the polynomial function of the 3rd order (cubic) fits the data very well.

# 11oder polynomial
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
#print(p1)
#PlotPolly(p1, x, y, 'highway-mpg')
# conclusion: We can conclude that the polynomial function of the 11th order fits the data very well.

from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2, include_bias=False)
pr
Z_pr=pr.fit_transform(Z)
print(Z.shape)  
print(Z_pr.shape)

#Pipelines

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
print(pipe)

Z = Z.astype(float)
pipe.fit(Z,y)

ypipe=pipe.predict(Z)
print(ypipe[0:4])

Input1 = [('scale', StandardScaler()), ('model', LinearRegression())]
pipe1 = Pipeline(Input1)
print(pipe1)
pipe1.fit(Z,y)
ypipe1=pipe1.predict(Z)
print(ypipe1[0:10])

#MSE and R^2 evaluation metrics
from sklearn.metrics import mean_squared_error

# Linear regression
lm.fit(X, Y)
print("The R^2 is: ", lm.score(X, Y))
Yhat=lm.predict(X)
print("The MSE is: ", mean_squared_error(usedcars_df['price'], Yhat))

# Multiple linear regression
lm.fit(Z, usedcars_df['price'])
print("The R^2 is: ", lm.score(Z, usedcars_df['price']))
Yhat=lm.predict(Z)
print("The MSE is: ", mean_squared_error(usedcars_df['price'], Yhat))

# Polynomial regression
from sklearn.metrics import r2_score
x = usedcars_df['highway-mpg']
y = usedcars_df['price']
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
yhat = p(x)
print("The R^2 is: ", r2_score(y, yhat))
print("The MSE is: ", mean_squared_error(usedcars_df['price'], yhat))   
print(mean_squared_error(usedcars_df['price'], yhat))
#conclusion: Based on the R^2 and MSE values, we can see that the multiple linear regression model performs better 
# than the simple linear regression and polynomial regression models.

#Prediction and Decision Making
new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
yhat=lm.predict(new_input)
yhat[0:5]
plt.plot(new_input, yhat)
#Conclusion: The plot shows the predicted price values for highway-mpg values ranging from 1 to 100.
