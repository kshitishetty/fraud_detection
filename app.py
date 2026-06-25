
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load and preprocess the dataset
df = pd.read_csv("creditcard.csv")
df = df.fillna(0)  # replaces NaNs with 0


X = df.drop("Class", axis=1)
y = df["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = [data.get(col, 0) for col in df.columns if col != "Class"]
    scaled_input = scaler.transform([input_features])
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][prediction]

    return jsonify({
        "fraud": int(prediction),
        "probability": round(prob * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
