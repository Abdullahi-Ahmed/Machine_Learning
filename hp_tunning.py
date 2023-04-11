# Import libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
data = pd.read_csv('iris.csv')

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(data.drop('species', axis=1), data['species'], test_size=0.3, random_state=42)

# Train a Naive Bayes classifier on the training set and evaluate its performance on the test set
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy on test set:', accuracy)

# Vary the size of the training set and observe how the performance of the classifier changes
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
test_accuracy = []
train_accuracy = []

for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(data.drop('species', axis=1), data['species'], test_size=1-size, random_state=42)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_accuracy.append(accuracy_score(y_train, y_pred_train))
    test_accuracy.append(accuracy_score(y_test, y_pred_test))

# Plot the training set size against the accuracy of the classifier on the test set
import matplotlib.pyplot as plt
plt.plot(train_sizes, train_accuracy, 'o-', color='blue', label='Training accuracy')
plt.plot(train_sizes, test_accuracy, 'o-', color='red', label='Test accuracy')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Set Size')
plt.legend()
plt.show()

# Repeat for different levels of model complexity
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
test_accuracy = []
train_accuracy = []

for alpha in alpha_values:
    clf = GaussianNB(var_smoothing=alpha)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_accuracy.append(accuracy_score(y_train, y_pred_train))
    test_accuracy.append(accuracy_score(y_test, y_pred_test))

# Plot the model complexity (alpha) against the accuracy of the classifier on the test set
plt.plot(alpha_values, train_accuracy, 'o-', color='blue', label='Training accuracy')
plt.plot(alpha_values, test_accuracy, 'o-', color='red', label='Test accuracy')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Model Complexity')
plt.legend()
plt.show()
