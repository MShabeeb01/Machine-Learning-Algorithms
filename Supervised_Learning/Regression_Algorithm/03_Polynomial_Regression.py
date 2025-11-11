# Import important libraries
from sklearn.linear_model import LinearRegression     # For linear regression model
from sklearn.preprocessing import PolynomialFeatures  # To create polynomial features
from sklearn.model_selection import train_test_split  # To split dataset into train & test
from sklearn.metrics import mean_squared_error        # To evaluate model performance
import numpy as np                                    # For numerical operations

# Sample Data (Years of Experience vs Salary)
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])   # Independent variable (Experience)
Y = np.array([45000,50000,60000,80000,110000,150000,200000,300000,400000,500000])  # Dependent variable (Salary)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Transform input features into polynomial form
# This allows the model to learn nonlinear relationships
poly = PolynomialFeatures()           
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize and train the Linear Regression model using polynomial features
model = LinearRegression()
model.fit(X_train_poly, Y_train)

# Predict salaries using the test data
Y_pred = model.predict(X_test_poly)

# Evaluate model performance using Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)     # Lower value means better model accuracy

# Print predicted and actual salary values
print("Predicted Values:", Y_pred)    # Salaries predicted by the model
print("Actual Values:", Y_test)       # Real salaries from test data


# 📘 NOTES 

#  What is Polynomial Regression?
# Polynomial Regression is an advanced form of Linear Regression where
# the model can learn curved (non-linear) relationships between X and Y.
# Instead of a straight line (y = mx + c), the equation looks like:
# y = a0 + a1*x + a2*x^2 + a3*x^3 + ... 
# It helps fit data that doesn't follow a straight line pattern.

#  What does PolynomialFeatures() do?
# It converts your simple input X into multiple power terms like:
# Example: if X = [2], polynomial features make it [1, 2, 4] (that is x^0, x^1, x^2)
# So it gives the model more power to learn curved patterns.

#  What does fit_transform() mean?
# - 'fit' learns how to transform the data (like finding mean, powers, etc.)
# - 'transform' actually applies that transformation to the data.
# So 'fit_transform()' = learn + apply transformation.

#  What does model.fit() mean?
# 'fit()' means the model is learning from the training data.
# It finds the best coefficients (weights) to minimize the prediction error.

#  What does model.predict() mean?
# After the model is trained, 'predict()' uses that learned pattern
# to make predictions on new (unseen) data.

#  What is Mean Squared Error (MSE)?
# MSE measures how close the predictions are to the actual values.
# Formula: average of (actual - predicted)^2
# Smaller MSE = better model accuracy.
