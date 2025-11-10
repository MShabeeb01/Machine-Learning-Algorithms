# Step 1: Import important libraries
from sklearn.linear_model import Ridge, Lasso        # For Ridge and Lasso regression
from sklearn.model_selection import train_test_split # To split data for training and testing
from sklearn.metrics import mean_squared_error       # To check how accurate the model is
import numpy as np                                   # For handling numerical data

# Step 2: Create sample data (House size vs Price)
# X = size of house (in square feet)
# Y = price of the house (in dollars)
X = np.array([[1400],[1600],[1700],[1875],[1100],[1550],[2350],[2450],[1420],[1700]])
Y = np.array([245000,312000,279000,308000,199000,219000,405000,324000,319000,255000])

# Step 3: Split data into training and testing parts
# The model will learn using training data and we’ll test it on testing data
# test_size=0.2 means 20% data will be for testing, 80% for training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Step 4: Ridge Regression
# Ridge adds a small penalty to reduce overfitting
# Overfitting means: model learns too much from training data and fails on new data
ridge_model = Ridge(alpha=1.0)     # alpha controls how strong the penalty is
ridge_model.fit(X_train, Y_train)  # Train the model
ridge_pred = ridge_model.predict(X_test)  # Predict house prices on test data

# Mean Squared Error (MSE) tells how far predictions are from actual values
ridge_mse = mean_squared_error(Y_test, ridge_pred)
print("Ridge Mean Squared Error:", ridge_mse)


# Step 5: Lasso Regression
# Lasso also reduces overfitting, but it can make some features (inputs) zero
# That means it automatically removes less important features
lasso_model = Lasso(alpha=0.1)     # alpha again controls the penalty strength
lasso_model.fit(X_train, Y_train)  # Train the model
lasso_pred = lasso_model.predict(X_test)  # Predict house prices on test data

lasso_mse = mean_squared_error(Y_test, lasso_pred)
print("Lasso Mean Squared Error:", lasso_mse)


# Step 6: What is the difference between Ridge and Lasso?
# Ridge Regression → Keeps all features, just reduces their effect.
# Lasso Regression → Can remove some unimportant features by making them zero.
# Both help the model perform better on new unseen data.


# Step 7: What is Mean Squared Error (MSE)?
# MSE = Average of (predicted value - actual value)^2
# Lower MSE means the predictions are closer to actual values → better model


# Step 8: Summary in very simple words
# - Ridge and Lasso are special types of linear regression.
# - They help prevent overfitting.
# - alpha is a number that decides how much the model should be controlled.
# - Smaller MSE means your model is predicting more accurately.
# - Lasso can remove unnecessary inputs automatically.


# Overfitting

# Overfitting means the model learns the training data too well.
# It memorizes data instead of learning patterns.

# Result:
# - Works very well on training data
# - Performs poorly on new/unseen (test) data

# Example:
# Like memorizing answers instead of understanding the concept.
# You’ll fail when the question changes slightly.

# Signs of Overfitting:
# - High accuracy on training data
# - Low accuracy on test data

# How to avoid Overfitting:
# 1. Use regularization (Ridge or Lasso)
# 2. Use more training data
# 3. Simplify the model
# 4. Use cross-validation
# 5. Stop training early if test error increases

# Simple summary:
# Overfitting → Model memorizes training data
# Underfitting → Model is too simple
# Good Fit → Learns patterns that work well on both training and test data
