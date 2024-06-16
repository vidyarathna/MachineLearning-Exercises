from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset (example: Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Initialize model (example: Logistic Regression)
model = LogisticRegression()

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

mean_accuracy = np.mean(accuracies)
print(f'Mean accuracy: {mean_accuracy}')
