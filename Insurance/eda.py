import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('Insurance/medical_insurance_dataset.csv')
#print(df.head(5))

headers=['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region', 'charges']
df.columns = headers
#print(df.head(5))

#print(df.isna().sum())
df.replace('?', np.nan, inplace=True)
print(df.isna().sum())

# Data Wrangling
df.info()
is_smoker = df['smoker'].value_counts().idxmax()
df.loc[:, 'smoker'] = df[['smoker']].replace(np.nan, is_smoker)
mean_age = df['age'].astype('float').mean(axis=0)
df.loc[:, 'age'] = df[['age']].replace(np.nan, mean_age)
most_frequent = df[['smoker']].mode()

print(df.head(5))
df[["age", "smoker"]] = df[["age", "smoker"]].astype("int")

print(df.info())
df[["chrages"]] = np.round(df[["charges"]], 2)
print(df.head())

#EDA
sns.regplot(x = "bmi",y = "charges", data=df, line_kws={"color" : "red"})
plt.ylim(0,)
plt.show()
plt.close()

sns.boxplot(x="smoker", y="charges", data=df)
plt.show()
plt.close()

print(df.corr())

# Model Development:
X=df[["smoker"]]
Y=df[["charges"]]
lr = LinearRegression()
lr.fit(X,Y)
print("The R^2 value of charges vs smoker :", lr.score(X,Y))

Z = df[["age", "gender","bmi", "no_of_children", "smoker", "region"]]
lr.fit(Z,Y)
print("The R^2 value of charges vs all other columns :", lr.score(Z,Y))

# pipeline
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
Z = Z.astype('float')
pipe.fit(Z, Y)
ypipe = pipe.predict(Z)
print(r2_score(Y, ypipe))

# Model Refinement: 
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=1)

RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train,y_train)
yhat=RidgeModel.predict(x_test)
print("R^2 : ", r2_score(y_test,yhat))

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
yhat_pr=RidgeModel.predict(x_test_pr)
print("R^2 of polynomial degree 2: ", r2_score(y_test,yhat_pr))