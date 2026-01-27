from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

df = pd.read_csv('laptop/laptop_pricing_dataset_mod2.csv', header=0)
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
df.info()

# Cross validation to imporve the model
y_data = df[['Price']]
x_data = df.drop(['Price'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.1, random_state=1)
print("number of test samples : ", x_test.shape[0])
print("number of training samples : ", x_train.shape[0])

lr = LinearRegression()
lr.fit(x_train[['CPU_frequency']], y_train)
lr.fit(x_test[['CPU_frequency']],y_test)

print("R^2 value for x_train ", lr.score(x_train[['CPU_frequency']], y_train))
print("R^2 value for y_train ", lr.score(x_test[['CPU_frequency']],y_test))

Rcross = cross_val_score(lr, x_data[['CPU_frequency']],y_data,cv=4)
print("Mean value of R^2 score ", Rcross.mean())
print("Standard deviation of R^2 score ", Rcross.std())

# Overfitting    
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=1)

lre=LinearRegression()
Rsqu_test = []
order = [1,2,3,4,5]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['CPU_frequency']])
    x_test_pr = pr.fit_transform(x_test[['CPU_frequency']])
    lre.fit(x_train_pr, y_train)
    Rsqu_test.append(lre.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.show()
plt.close() 

# Ridge Regression:
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['CPU_frequency','RAM_GB', 'CPU_core','OS','GPU','Category']])
x_test_pr = pr.fit_transform(x_test[['CPU_frequency','RAM_GB', 'CPU_core','OS','GPU','Category']])

Rsqu_test = []
Rsqu_train = []
Alpha = np.arange(0.001,1,0.001)
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha)
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)
    
# R^2 values
plt.figure(figsize=(10, 6))
plt.plot(Alpha, Rsqu_test, label='validation data')
plt.plot(Alpha,  Rsqu_train, 'r', label='training data')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.ylim(0,1)
plt.legend()
plt.show()
plt.close()

#GridSearchCV
parameters1 = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}

RR = Ridge()

Grid1 = GridSearchCV(RR,parameters1,cv=4)
Grid1.fit(x_data[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']],y_data)

BestRR=Grid1.best_estimator_
print(BestRR.score(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU','Category']], y_test))