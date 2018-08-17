# Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# Load the data
# data on the average life expectancy at birth and
# the average BMI for males across the world. The data comes from Gapminder.
# https://www.gapminder.org/
bmi_life_data = pd.read_csv("./data/bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
#Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Make a prediction using the model
# Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)
