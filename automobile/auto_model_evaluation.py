import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm
warnings.filterwarnings('ignore')

df = pd.read_csv('automobile/module_5_auto.csv',header=0)

df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
print(df.head())
print(df.info())

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()
    
def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    # training data
    xmax = np.max([xtrain.values.max(), xtest.values.max()])
    xmin = np.min([xtrain.values.min(), xtest.values.min()])
    x=np.arange(xmin, xmax, 0.1)
    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    plt.close()
    
y_data = df['price']
x_data = df.drop('price', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

x_train1,x_test1,y_train1,y_test1 = train_test_split(x_data, y_data, test_size=0.40, random_state=0)
print("number of test samples :", x_test1.shape[0])
print("number of training samples:",x_train1.shape[0])

# Linear Regression Model
lre=LinearRegression()
lre.fit(x_train[['horsepower']], y_train)

# R^2 for x_train
print("R^2 for training set:", lre.score(x_train[['horsepower']], y_train))
# R^2 for x_test
print("R^2 for test set:", lre.score(x_test[['horsepower']], y_test))
# R^2 is smaller on the test set, this is an indication of overfitting

#R^2 for x_train1
print("R^2 for training set:", lre.score(x_train1[['horsepower']], y_train1))
# R^2 for x_test1
print("R^2 for test set:", lre.score(x_test1[['horsepower']], y_test1))
# R^2 is smaller on the test set, this is an indication of overfitting

# Cross validation Score
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print("Cross validation scores:", Rcross)
print("Mean of cross validation scores:", Rcross.mean())
print("Standard deviation of cross validation scores:", Rcross.std())
# We can see that the mean of the cross-validation score is very close to the R^2 value for the test set
# negative mean squared error
MSE = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4, scoring='neg_mean_squared_error')
MSE = -1 * MSE
print("Mean Squared Error:", MSE)

# Average R^2 
Rcross2 = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)
print("Average R^2 (2-folds):", Rcross2.mean())

Yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
print(Yhat[0:5])

# Overfitting, underfitting and model selection
lr = LinearRegression()
lr.fit(x_train[['horsepower','curb-weight','engine-size','highway-mpg']], y_train)
yhat_train = lr.predict(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
yhat_train[0:5]
yhat_test = lr.predict(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])
yhat_test[0:5]

# Distribution plot for training data
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

Title = 'Distribution  Plot of  Predicted Value Using Test Data vs Test Data Distribution'
DistributionPlot(y_test, yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)
x_train.info()
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
print(pr)

poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)
print(yhat[0:5])
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)

# Conclusions:
# 1. In the case of the training data, the predicted values follow the distribution of the actual values very closely.
# 2. In the case of the test data, the predicted values do not follow the distribution of the actual values as closely as in the training data.
# This is an indication that the model is overfitting the training data.
# 3. We can conclude that as the complexity of the model increases, the training data fits better, but the test data fits worse.    

# R^2 for different polynomial degrees
Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    Rsqu_test.append(poly.score(x_test_pr, y_test))
    print(Rsqu_test)

plt.plot(order, Rsqu_test)
plt.xlabel('order of polynomial')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data vs Polynomial Order')
plt.text(3, 0.75, 'Maximum R^2 ')
plt.show()
plt.close()

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)

#interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05));

pr1=PolynomialFeatures(degree=2)
x_train_pr1=pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1=pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(x_train_pr1.shape)
print(x_test_pr1.shape)

# Linearregression
poly1 = LinearRegression()
poly1.fit(x_train_pr1, y_train)
yhat_test1=poly1.predict(x_test_pr1)

Title = 'Distribution  Plot of  Predicted Value Using Test Data vs Test Data Distribution'
DistributionPlot(y_test, yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)
# Conclusion:
# 1. In this case, the predicted values follow the distribution of the actual values very closely even though we are using the test data.
# 2. We can conclude that by increasing the number of features in the model, we can reduce the overfitting problem that we had before.  

# Ridge Regression:
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

RidgeModel=Ridge(alpha=1)
RidgeModel.fit(x_train_pr, y_train)
yhat=RidgeModel.predict(x_test_pr)
print('predicted:', yhat[0:4])
print('test data :', y_test[0:4].values)

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha =  10 * np.array(range(0,1000))
pbar = tqdm(Alpha)
for alpha in pbar:
    RidgeModel = Ridge(alpha=alpha)
    RidgeModel.fit(x_train_pr, y_train)
    test_score, train_score = RidgeModel.score(x_test_pr, y_test), RidgeModel.score(x_train_pr, y_train)
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

# R^2 plot
width = 12
height = 10
plt.figure(figsize=(width, height))
plt.plot(Alpha, Rsqu_test, label='validation data R^2')
plt.plot(Alpha, Rsqu_train, label='training data R^2')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.title('Ridge Regression: R^2 Using Training and Validation Data vs Alpha')
plt.show()
plt.close()

# Ridge Model
RidgeModel = Ridge(alpha=10)
RidgeModel.fit(x_train_pr, y_train)
RidgeModel.score(x_test_pr, y_test)

# Grid Search CV
parameters1 = [{'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
parameters1
RR=Ridge()
RR

Grid1 = GridSearchCV(RR, parameters1, cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

BestRR=Grid1.best_estimator_
print(BestRR)  
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

parameters2 = [{'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}]

Grid2 = GridSearchCV(RR, parameters2, cv=4)
Grid2.fit(x_scaled, y_data)

best_ridge_model = Grid2.best_estimator_
best_alpha = Grid2.best_params_['alpha']
print(best_ridge_model)
print("Best alpha:", best_alpha)
