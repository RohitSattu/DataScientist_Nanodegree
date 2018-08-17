# Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# Load the data
train_data = pd.read_csv('./data/reg_data.csv')
# print(train_data) # test
X = train_data.iloc[:,:-1]
# print(X) # test
y = train_data.iloc[:,-1]
# print(y) # test
# Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# Fit the model.
lasso_reg.fit(X, y)

# Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)

# sklearn's Lasso class to fit a linear regression model to the data,
# while also using L1 regularization to control for model complexity.
