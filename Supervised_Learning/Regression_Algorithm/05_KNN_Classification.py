#  K-Nearest Neighbors (KNN)  (Hours Studied & Grades vs Pass/Fail)

# Important Libraries 
from sklearn.neighbors import KNeighborsClassifier       # For KNN algorithm
from sklearn.model_selection import train_test_split     # To split dataset
from sklearn.metrics import accuracy_score, confusion_matrix  # For model evaluation
import numpy as np                                       # For numerical operations

# 🧠 Sample Data
# X --> Independent variables (Hours studied, Previous Grades)
# Y --> Dependent variable (Pass/Fail)
# 0 = Fail, 1 = Pass
X = np.array([[1,50],[2,60],[3,55],[4,65],[5,70],[6,75],[7,80],[8,85],[9,90],[10,100]])
Y = np.array([0,0,0,0,1,1,1,1,1,1])

#  Split the data into training and testing sets (80% training, 20% testing)
# random_state ensures reproducibility (same split each time)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 🧩 Initialize and train the KNN model
# n_neighbors=3 means the model looks at the 3 nearest neighbors to classify new data
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, Y_train)     # 'fit' means training the model using the given data

# 🔮 Make Predictions using the test data
Y_pred = model.predict(X_test)  # 'predict' uses trained model to classify unseen data

# 📊 Evaluate Model Performance
accuracy = accuracy_score(Y_test, Y_pred)        # How many predictions are correct
conf_matrix = confusion_matrix(Y_test, Y_pred)   # To visualize prediction errors

# 📈 Display Results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Predicted Values:", Y_pred)
print("Actual Values:", Y_test)


# 📘 NOTES (Beginner Friendly)

# 🧠 What is K-Nearest Neighbors (KNN)?
# KNN is a simple and powerful **classification algorithm**.
# It classifies a new data point based on how its neighbors (nearby points) are labeled.
# For example, if you want to know whether a student will pass or fail,
# KNN looks at the K closest students (based on study hours and grades)
# and assigns the most common class among them.

# The algorithm works in these steps:
# 1️⃣ Choose the number of neighbors (K).
# 2️⃣ Calculate the distance between the new point and all training points.
# 3️⃣ Select the K nearest data points.
# 4️⃣ Assign the class (0 or 1) that appears most often among those neighbors.

# Example:
# If K=3 and among the 3 nearest neighbors, 2 students passed and 1 failed,
# the new student will be classified as "Pass".

# ⚙️ What does model.fit() do?
# 'fit()' means the model stores (memorizes) the training data points and their labels.
# KNN doesn’t really “learn” like other models; it just keeps the data to compare later.

# 🔮 What does model.predict() do?
# 'predict()' checks new (test) data points against all training data,
# finds the K closest neighbors, and assigns the most frequent label among them.

# 📈 What is Accuracy?
# Accuracy tells how many of the model’s predictions were correct.
# Formula: (Number of correct predictions / Total predictions)
# Example: If 8 out of 10 are correct, accuracy = 0.8 or 80%

#  What is a Confusion Matrix?
# It shows how well the model performed in terms of True/False predictions:
# [[True Negatives, False Positives],
#  [False Negatives, True Positives]]
#
# - True Negative (TN): Model correctly predicted Fail
# - True Positive (TP): Model correctly predicted Pass
# - False Negative (FN): Model predicted Fail but actual was Pass
# - False Positive (FP): Model predicted Pass but actual was Fail

# Extra Tip:
# Choosing the right 'K' value is important:
# - Small K (like 1 or 2): model becomes sensitive to noise (overfitting)
# - Large K: model becomes too generalized (underfitting)
# Common practice: try different K values and pick the one with best accuracy.
