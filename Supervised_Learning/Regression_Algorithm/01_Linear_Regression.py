#Import important libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#Sample Data (House size vs House Price)
X = np.array([[1400],[1600],[1700],[1875],[1100],[1550],[2350],[2450],[1420],[1700]])
Y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000 ])

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Initialize and Train the Model
model = LinearRegression()
model.fit(X_train,y_train)

#Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:", mse)
print("Predicted Values:", y_pred)
print("Actual Values:",y_test)


# 🧠 Understanding Train-Test Split


# In machine learning, we divide our data into two parts:
# 1️⃣ Training Data (used to teach the model)
# 2️⃣ Testing Data (used to check how well the model learned)

# train_test_split() automatically divides the dataset for us.
# Example:
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#  test_size=0.2 means:
#     - 20% of the data will be used for testing (unseen by the model)
#     - The remaining 80% will be used for training (to fit or learn the pattern)

# During training:
#     The model learns the relationship between input (X_train) and output (y_train)
#     It calculates a best-fit line (Y = mX + c)

# During testing:
#     The model predicts prices for new data (X_test) that it never saw before
#     These predictions are stored in y_pred

# So, in this project:
#     ✅ 80% of the house size–price data was used to train the model
#     ✅ 20% of the data was used to test how well it can predict unseen house prices




# 📊 Mean Squared Error (MSE)

# MSE tells us how far our predictions are from the actual values, on average.
# Formula:
#     MSE = (1/n) * Σ (y_actual - y_predicted)²

# Lower MSE → Better model (predictions are close to actual)
# Higher MSE → Worse model (predictions are far from actual)

# Important: The error is "squared", so large mistakes affect the MSE a lot.
# Example:
#     If the model predicts 10,000 too high, (10,000)² = 100,000,000 gets added to the error!
