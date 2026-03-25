import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Commented out due to environment compatibility; matplotlib used instead
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure directories exist
os.makedirs('eda', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Import required libraries (Done above)

# 2. Load dataset
print("Loading dataset...")
df = pd.read_csv('weatherAUS.csv')

# 3. Data Preprocessing
print("Pre-processing data...")
# Handling missing values
df = df.dropna()

# Encoding categorical data
le = LabelEncoder()
categorical_cols = ['RainToday', 'RainTomorrow', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Location']
for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Feature selection (Based on requirements: Temp, Humidity, Wind speed, Pressure)
# We will use the 3pm versions as they are more indicative of tomorrow's rain
features = ['Temp3pm', 'Humidity3pm', 'WindSpeed3pm', 'Pressure3pm', 'RainToday']
X = df[features]
y = df['RainTomorrow']

# 4. Exploratory Data Analysis (EDA)
print("Performing EDA...")
plt.figure(figsize=(12, 6))
plt.plot(df.index[:100], df['Temp3pm'].head(100), label='Temperature (3pm)', color='orange')
plt.plot(df.index[:100], df['Humidity3pm'].head(100), label='Humidity (3pm)', color='blue')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Temperature and Humidity Trends')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('eda/trends.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(df['Rainfall'], bins=30, color='skyblue', edgecolor='black')
plt.title('Rainfall Distribution')
plt.xlabel('Rainfall')
plt.ylabel('Frequency')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.savefig('eda/rainfall_dist.png')
plt.close()

# 5. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train ML model
print("Training models...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression()

rf_model.fit(X_train_scaled, y_train)
lr_model.fit(X_train_scaled, y_train)

# Compare accuracy
rf_pred = rf_model.predict(X_test_scaled)
lr_pred = lr_model.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, rf_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# 7. Evaluate model (using the best one, usually RF)
best_model = rf_model if rf_acc > lr_acc else lr_model
best_name = "Random Forest" if rf_acc > lr_acc else "Logistic Regression"
y_pred = rf_pred if rf_acc > lr_acc else lr_pred

print(f"\nEvaluating Best Model: {best_name}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save the best model using pickle
print("Saving model...")
model_data = {
    'model': best_model,
    'scaler': scaler,
    'features': features
}
with open('models/weather_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Pipeline completed successfully!")
