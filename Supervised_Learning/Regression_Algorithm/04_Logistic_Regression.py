# Logistic Regression Example (Hours Studied vs Pass/Fail)

# Importing important libraries
from sklearn.linear_model import LogisticRegression      # For Logistic Regression model
from sklearn.model_selection import train_test_split     # To split data into train/test
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error  # For model evaluation
import numpy as np                                       # For numerical operations

#  Sample Data
# X --> Independent variable (Hours studied)
# Y --> Dependent variable (Pass/Fail)
# 0 = Fail, 1 = Pass
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
Y = np.array([0,0,0,0,1,1,1,1,1,1])

#  Split data into training and testing sets (80% training, 20% testing)
# random_state ensures the same data split every time you run the code
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#  Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)   # 'fit' means the model learns from training data

# Make Predictions using the test data
Y_pred = model.predict(X_test)   # 'predict' means use the trained model to predict outputs

#  Evaluate the Model Performance

#  Accuracy: measures how many predictions were correct
accuracy = accuracy_score(Y_test, Y_pred)

# Confusion Matrix: shows counts of correct and incorrect predictions
# Format: [[TrueNegatives, FalsePositives],
#          [FalseNegatives, TruePositives]]
conf_matrix = confusion_matrix(Y_test, Y_pred)

#  Mean Squared Error (MSE): measures the average squared difference
# between actual (Y_test) and predicted (Y_pred)
mse = mean_squared_error(Y_test, Y_pred)

# 📈 Display Results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Mean Squared Error:", mse)
print("Predicted Values:", Y_pred)
print("Actual Values:", Y_test)


# 📘 NOTES

#  What is Logistic Regression?
# Logistic Regression is a classification algorithm used when the output (Y) is categorical.
# It predicts probabilities of belonging to a particular class (like Pass/Fail or 0/1).
# Instead of fitting a straight line (like Linear Regression),
# it fits an 'S-shaped curve' called the Sigmoid Function.

#  What does model.fit() do?
# 'fit()' trains the model — it learns patterns between input (X) and output (Y).

#  What does model.predict() do?
# 'predict()' uses the trained model to make predictions on new/unseen data.

#  What is Accuracy?
# Accuracy = (Number of correct predictions / Total predictions) × 100
# It shows how well the model is performing overall.

#  What is a Confusion Matrix?
# It breaks down predictions into 4 categories:
# - True Positive (TP): Model correctly predicted 1 (Pass)
# - True Negative (TN): Model correctly predicted 0 (Fail)
# - False Positive (FP): Model predicted Pass but actual was Fail
# - False Negative (FN): Model predicted Fail but actual was Pass

#  What is Mean Squared Error (MSE)?
# Though MSE is more common in regression problems,
# it can still show how far predicted values are from actual ones.
# Lower MSE means the model’s predictions are closer to real values.
