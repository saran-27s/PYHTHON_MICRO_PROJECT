import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the model and scaler
print("Loading the pre-trained model and scaler...")
rf_model = joblib.load('models/ethereum_fraud_model.pkl')
scaler = joblib.load('models/ethereum_fraud_scaler.pkl')

# Load the wallet data for prediction
print("Loading wallet data for prediction...")
wallet_data = pd.read_csv('csv_files/wallet_data.csv')

# Display basic information about the data
print(f"Number of wallets for prediction: {len(wallet_data)}")
print("\nWallet data preview:")
print(wallet_data.head())

# Handle missing values (if any)
wallet_data = wallet_data.fillna(0)

# If the wallet data contains the Address column, save it for reference
if 'Address' in wallet_data.columns:
    addresses = wallet_data['Address']
    X_predict = wallet_data.drop(['Address'], axis=1)
else:
    X_predict = wallet_data.copy()

# If the wallet data contains the FLAG column (for evaluation)
has_true_labels = False
if 'FLAG' in X_predict.columns:
    y_true = X_predict['FLAG']
    X_predict = X_predict.drop(['FLAG'], axis=1)
    has_true_labels = True

# Ensure that the prediction data has the same features as the training data
model_features = [
    'Avg min between sent tnx', 'Avg min between received tnx', 
    'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx', 
    'Unique Received From Addresses', 'Unique Sent To Addresses', 
    'max value received', 'avg val received', 'max val sent', 
    'avg val sent', 'total Ether sent', 'total ether received', 
    'total ether balance'
]
missing_cols = set(model_features) - set(X_predict.columns)
extra_cols = set(X_predict.columns) - set(model_features)

if missing_cols:
    print(f"Warning: Missing columns in prediction data: {missing_cols}")
    for col in missing_cols:
        X_predict[col] = 0  # Add missing columns with default values

if extra_cols:
    print(f"Warning: Extra columns in prediction data: {extra_cols}")
    X_predict = X_predict.drop(columns=extra_cols)  # Remove extra columns

# Reorder columns to match the order used during training
X_predict = X_predict[model_features]

# Scale the features
X_predict_scaled = scaler.transform(X_predict)

# Make predictions
print("\nMaking predictions...")
y_pred = rf_model.predict(X_predict_scaled)
pred_proba = rf_model.predict_proba(X_predict_scaled)[:, 1]  # Probability of fraud

# Create a DataFrame with the predictions
results = pd.DataFrame({
    'Predicted_Fraud_Flag': y_pred,
    'Fraud_Probability': pred_proba
})

# Add the addresses if available
if 'addresses' in locals():
    results['Address'] = addresses

# Display the prediction results
print("\nPrediction Results:")
print(results.head(10))

# Save the predictions to a CSV file
results.to_csv('csv_files/fraud_prediction_results.csv', index=False)
print("Predictions saved to 'csv_files/fraud_prediction_results.csv'")

# Evaluate the model if true labels are available
if has_true_labels:
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nModel Accuracy on wallet_data.csv: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix on Wallet Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('images/wallet_data_confusion_matrix.png')
    plt.show()

# Visualize the distribution of fraud probabilities
plt.figure(figsize=(10, 6))
sns.histplot(results['Fraud_Probability'], bins=50, kde=True)
plt.title('Distribution of Fraud Probabilities')
plt.xlabel('Fraud Probability')
plt.ylabel('Count')
plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
plt.legend()
plt.tight_layout()
plt.savefig('images/fraud_probability_distribution.png')
plt.show()

# Visualize the predictions
plt.figure(figsize=(8, 6))
fraud_count = sum(y_pred)
non_fraud_count = len(y_pred) - fraud_count
plt.bar(['Non-Fraud (0)', 'Fraud (1)'], [non_fraud_count, fraud_count])
plt.title('Prediction Results')
plt.ylabel('Count')
for i, v in enumerate([non_fraud_count, fraud_count]):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.tight_layout()
plt.savefig('images/prediction_results.png')
plt.show()

print("Prediction program completed successfully!")