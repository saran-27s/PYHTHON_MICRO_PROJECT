import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('csv_files/train_data.csv')

# Check for missing values
print("Missing values in dataset:")
print(df.isnull().sum())

# Handle missing values (if any)
df = df.fillna(0)

# Prepare features and target variable
X = df.drop(['FLAG', 'Address'], axis=1)
y = df['FLAG']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, 'models/ethereum_fraud_scaler.pkl')

# Train the random forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(rf_model, 'models/ethereum_fraud_model.pkl')
print("Model saved as 'models/ethereum_fraud_model.pkl'")

# Get feature importances
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("\nFeature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {features[indices[f]]} ({importances[indices[f]]:.4f})")

# Visualizations
plt.figure(figsize=(12, 6))
plt.title('Feature Importances for Ethereum Fraud Detection')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('images/feature_importances.png')
plt.show()

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('images/confusion_matrix.png')
plt.show()

# Visualize distribution of transaction counts by fraud/non-fraud
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='FLAG', y='Sent tnx', data=df)
plt.title('Sent Transactions by Fraud Flag')

plt.subplot(1, 2, 2)
sns.boxplot(x='FLAG', y='Received Tnx', data=df)
plt.title('Received Transactions by Fraud Flag')
plt.tight_layout()
plt.savefig('images/transaction_distributions.png')
plt.show()

print("Training program completed successfully!")