import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Load the laptop_pricing_dataset_mod2 dataset
df = pd.read_csv('laptop/laptop_pricing_dataset_mod2.csv',header=0)
print(df.head())
print(df.info())

# Single linear regression model
X = df[['CPU_frequency']]
y = df['Price']
lm=LinearRegression()
lm.fit(X,y)
yhat=lm.predict(X)
print("Single Linear Regression Model")
print("Predicted values:", yhat[0:5])

# Distribution plot predicted vs actual
ax1 = sns.displot(y, color="r", label="Actual Value")
sns.displot(yhat, color="b", label="Fitted Values")
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Laptops')
#plt.show()
#plt.close()

# Mean Squared Error and R2
mse = mean_squared_error(y, yhat)
r2_value = lm.score(X, y)
print("Mean Squared Error:", mse)
print("R2 Score:", r2_value)

# Multiple linear regression model
Z = df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS', 'GPU', 'Category']]
lm2=LinearRegression()
lm2.fit(Z,y)
yhat2=lm2.predict(Z)
print("Multiple Linear Regression Model")
print("Predicted values:", yhat2[0:5])

#Distribution plot predicted vs actual
ax2 = sns.displot(y, color="r", label="Actual Value")
sns.displot(yhat2, color="b", label="Fitted Values")
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Laptops')
plt.show()
plt.close()    

# R2 and MSE
mse2 = mean_squared_error(y, yhat2)
r2_score2 = lm2.score(Z, y)
print("Mean Squared Error:", mse2)
print("R2 Score:", r2_score2)

#Conclusions: 
#1. The multiple linear regression model performs better than the single linear regression model, 
# as indicated by the higher R2 score and lower MSE.
#2. The distribution plots show that the predicted values from the multiple linear regression model 
# are closer to the actual values compared to the single linear regression model.
#3. This suggests that considering multiple features leads to a more accurate prediction of laptop prices.  

# Polynomial regression
x = df[['CPU_frequency']]
y = df['Price']
f1 = np.polyfit(x['CPU_frequency'], y, 1)
p1 = np.poly1d(f1)
print("Polynomial Regression Model")
print(p1)

f3 = np.polyfit(x['CPU_frequency'], y, 3)
p3 = np.poly1d(f3)
print(p3)

f5 = np.polyfit(x['CPU_frequency'], y, 5)
p5 = np.poly1d(f5)
print(p5)
#conclusion: We can see that as the degree of the polynomial increases, the polynomial function fits the data better.

def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(independent_variable.min(), independent_variable.max(), 100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ ' + Name)
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.xlabel(Name)
    plt.ylabel('Price of Laptops')
    
    plt.show()
    plt.close()
# call the function
PlotPolly(p1, x['CPU_frequency'], y, 'CPU_frequency')
PlotPolly(p3, x['CPU_frequency'], y, 'CPU_frequency')
PlotPolly(p5, x['CPU_frequency'], y, 'CPU_frequency')
# conclusion: We can conclude that the polynomial function of the 5th order fits the data very well.

# R2 and MSE for polynomial regression
yhat_p1 = p1(x['CPU_frequency'])
yhat_p3 = p3(x['CPU_frequency'])
yhat_p5 = p5(x['CPU_frequency'])

mse_p1 = mean_squared_error(y, yhat_p1)
mse_p3 = mean_squared_error(y, yhat_p3)
mse_p5 = mean_squared_error(y, yhat_p5)     

r2_p1 = r2_score(y, yhat_p1)
r2_p3 = r2_score(y, yhat_p3)
r2_p5 = r2_score(y, yhat_p5)
print("Polynomial Degree 1 - MSE:", mse_p1, "R2:", r2_p1)
print("Polynomial Degree 3 - MSE:", mse_p3, "R2:", r2_p3)
print("Polynomial Degree 5 - MSE:", mse_p5, "R2:", r2_p5)
# conclusion: As the degree of the polynomial increases, the R2 value increases and the MSE decreases, 
# indicating a better fit to the data.  

# Pipelines
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline as pipeline
input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = pipeline(input)
Z = Z.astype(float)
pipe.fit(Z, y)
yhat_pipe = pipe.predict(Z)
print("Pipeline Predicted values:", yhat_pipe[0:5])

# R2 and MSE for pipeline
mse_pipe = mean_squared_error(y, yhat_pipe)
r2_pipe = r2_score(y, yhat_pipe)
print("Pipeline Mean Squared Error:", mse_pipe)
print("Pipeline R2 Score:", r2_pipe)