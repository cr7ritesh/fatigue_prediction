# Fatigue Prediction using Synthetic Data

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Generate synthetic data 
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'heart_rate': np.random.normal(75, 10, n_samples),
    'skin_temp': np.random.normal(33, 1, n_samples),
    'eda': np.random.normal(0.5, 0.1, n_samples),  # electrodermal activity
    'activity_level': np.random.uniform(0, 10, n_samples),
    'sleep_duration': np.random.normal(7, 1.5, n_samples),
    'steps': np.random.randint(1000, 15000, n_samples),
    'fatigue_level': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])   # 0: Low, 1: High
})

# Define feature and target
features = ['heart_rate', 'skin_temp', 'eda', 'activity_level', 'sleep_duration', 'steps']
target = 'fatigue_level'
X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model on all features to get importances
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Get top 4 important features
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_imp_df.sort_values("Importance", ascending=False, inplace=True)
selected_features = feat_imp_df.head(4)["Feature"].values

# Filter data for selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train final model
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train_selected, y_train)

# Evaluate
y_pred = final_model.predict(X_test_selected)
print("Selected Features:", selected_features)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature importance of selected features
selected_importances = final_model.feature_importances_
selected_feat_imp_df = pd.DataFrame({"Feature": selected_features, "Importance": selected_importances})
selected_feat_imp_df.sort_values("Importance", ascending=False, inplace=True)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=selected_feat_imp_df)
plt.title("Feature Importances")
plt.show()