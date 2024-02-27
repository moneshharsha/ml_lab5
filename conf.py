import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_excel(r"C:\\Users\\mones\\OneDrive\\Documents\\Traing_Data_614.xlsx")

# Preprocess the text data
data['Final answer'] = data['Final answer'].map({'correct': 1, 'incorrect': 0})

# Remove rows with NaN values in the target variable
data = data.dropna(subset=['Final answer'])

# Split the dataset into features and target variable
X = data[['model_answer', 'student_answer']]
y = data['Final answer']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['model_answer'] + ' ' + X_train['student_answer'])
X_test_vectorized = vectorizer.transform(X_test['model_answer'] + ' ' + X_test['student_answer'])

# Train a kNN classifier
k = 3  # Define the number of neighbors
kNN_classifier = KNeighborsClassifier(n_neighbors=k)
kNN_classifier.fit(X_train_vectorized, y_train)

# A1. Evaluate confusion matrix and other performance metrics
# Confusion matrix and performance metrics for training data
conf_matrix_train = confusion_matrix(y_train, kNN_classifier.predict(X_train_vectorized))
precision_train = precision_score(y_train, kNN_classifier.predict(X_train_vectorized))
recall_train = recall_score(y_train, kNN_classifier.predict(X_train_vectorized))
f1_train = f1_score(y_train, kNN_classifier.predict(X_train_vectorized))

# Confusion matrix and performance metrics for test data
conf_matrix_test = confusion_matrix(y_test, kNN_classifier.predict(X_test_vectorized))
precision_test = precision_score(y_test, kNN_classifier.predict(X_test_vectorized))
recall_test = recall_score(y_test, kNN_classifier.predict(X_test_vectorized))
f1_test = f1_score(y_test, kNN_classifier.predict(X_test_vectorized))

# Print performance metrics
print("Training Confusion Matrix:")
print(conf_matrix_train)
print("Training Precision:", precision_train)
print("Training Recall:", recall_train)
print("Training F1-Score:", f1_train)

print("\nTest Confusion Matrix:")
print(conf_matrix_test)
print("Test Precision:", precision_test)
print("Test Recall:", recall_test)
print("Test F1-Score:", f1_test)

# A2. Calculate MSE, RMSE, MAPE, and R2 scores
# Assuming y_true and y_pred are available
# Let's say we have y_true and y_pred calculated from a regression model
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 3.1, 4.2, 5.3])

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nMean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Percentage Error:", mape)
print("R2 Score:", r2)



# A3. Generate training data with 2 features and 2 classes
# Assuming you have 20 data points with 2 features (X and Y) and 2 classes (class0 and class1)
# Placeholder code
np.random.seed(0)
X_train = np.random.uniform(1, 10, (20, 2))
y_train = np.random.randint(0, 2, 20)

# Plot the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Training Data with 2 Features and 2 Classes')
plt.show()

# A4. Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1
# This creates a test set of about 10,000 points
# Placeholder code
X_test = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
X_test = np.array(X_test).reshape(-1, 2)

# Classify these points with the above training data using kNN classifier (k = 3)
# Placeholder code
y_test_pred = kNN_classifier.predict(X_test)

# Make a scatter plot of the test data output with test points colored as per their predicted class colors
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap=plt.cm.coolwarm)
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Test Data Output with 2 Features and 2 Classes')
plt.show()

# A5. Repeat A4 exercise for various values of k and observe the change in the class boundary lines
# Placeholder code