# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset from sklearn
from sklearn.datasets import load_iris
data = load_iris()

# Prepare the features (X) and target labels (y)
X = data.data  # Features (flower measurements)
y = data.target  # Labels (flower types)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the K-Nearest Neighbors classifier with k=3
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the model by calculating the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict the class for new data (example flower measurements)
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example new data (measurements)
prediction = knn.predict(new_data)

# Output the predicted class
print(f"Predicted class: {data.target_names[prediction][0]}")
