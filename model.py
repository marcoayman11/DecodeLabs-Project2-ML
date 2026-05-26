# =========================
# AI Classification Project
# =========================

# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================
# Load Dataset
# =========================

iris = load_iris()

X = iris.data       # Features
y = iris.target     # Labels

# =========================
# Split Dataset
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Create Model
# =========================

model = KNeighborsClassifier(n_neighbors=3)

# =========================
# Train Model
# =========================

model.fit(X_train, y_train)

# =========================
# Make Predictions
# =========================

predictions = model.predict(X_test)

# =========================
# Evaluate Model
# =========================

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# =========================
# Show Predictions
# =========================

print("\nPredictions:")
print(predictions)

print("\nActual Values:")
print(y_test)
