import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Task A3: Generate training data
# Assuming your Excel file is named 'project_data.xlsx' and contains columns 'Feature1', 'Feature2', and 'Class'
df = pd.read_excel('project_data.xlsx')
X_train = df[['Feature1', 'Feature2']].values
y_train = df['Class'].values

# Scatter plot for training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1')
plt.title('Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Task A4: Generate test data
x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
X_test = np.array([[x, y] for x in x_values for y in y_values])

# Classify test data using kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_test = knn.predict(X_test)

# Scatter plot for test data with predicted classes
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_pred_test == 0][:, 0], X_test[y_pred_test == 0][:, 1], color='blue', label='Predicted Class 0')
plt.scatter(X_test[y_pred_test == 1][:, 0], X_test[y_pred_test == 1][:, 1], color='red', label='Predicted Class 1')
plt.title('Test Data with Predicted Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Task A5: Repeat A4 for various values of k
k_values = [1, 3, 5, 7]  # Example values of k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_test = knn.predict(X_test)
    
    # Scatter plot for test data with predicted classes for current k
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[y_pred_test == 0][:, 0], X_test[y_pred_test == 0][:, 1], color='blue', label='Predicted Class 0')
    plt.scatter(X_test[y_pred_test == 1][:, 0], X_test[y_pred_test == 1][:, 1], color='red', label='Predicted Class 1')
    plt.title(f'Test Data with Predicted Classes (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
