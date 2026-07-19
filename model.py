import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load cleaned data
df = pd.read_csv('diabetes_clean.csv')

# Separate features (X) and label (y)
X = df.drop('Outcome', axis=1)   # everything except the answer
y = df['Outcome']                 # the answer (diabetic or not)

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the features (makes model more accurate)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and scaler for the web app
joblib.dump(model,  'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print('Model saved!')