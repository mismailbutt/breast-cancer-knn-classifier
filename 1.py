
# Step 1: Import the libraries we need
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Step 2: Load the dataset (569 samples, 2 classes, 30 features)
data = load_breast_cancer()
X = data.data          # features: 30 measurements of the tumor (size, texture, etc.)
y = data.target        # labels: 0 = malignant (harmful), 1 = benign (harmless)

print("Dataset shape:", X.shape)
print("Classes:", data.target_names)

# Step 3: Look at the data (understand it) - print the first 5 rows
print("\nFirst 5 samples (first 5 features only):\n", X[:5, :5])
print("Their labels:", y[:5])

# Step 4: Train-Test Split (80% training, 20% testing)
# random_state is fixed so the result is the same every time we run it
# stratify=y keeps both classes in the same proportion in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 5: Feature Scaling (the "gatekeeper" rule)
# KNN works using distance, so all features need to be on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn the scale from training data
X_test_scaled = scaler.transform(X_test)          # apply the same scale to test data

# Step 6: Build and train the model (Instantiate -> Fit)
model = KNeighborsClassifier(n_neighbors=5)   # K=5 (majority vote from 5 nearest neighbors)
model.fit(X_train_scaled, y_train)

# Step 7: Get predictions on the test data
predictions = model.predict(X_test_scaled)

# Step 8: Output Validation - don't trust accuracy alone
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)

# Confusion Matrix - shows exactly where the model got things right or wrong
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Precision, Recall, F1-score - a more detailed report
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=data.target_names))

# Step 9 (Bonus): Predict one test sample and check it
sample = [X_test[0]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("\nSample prediction:", data.target_names[prediction[0]])
print("Actual label was:", data.target_names[y_test[0]])
