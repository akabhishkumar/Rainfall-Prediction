from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model and tools [cite: 1222, 1233]
data = joblib.load('aussie_rain.joblib')
model = data['model']
imputer = data['imputer']
scaler = data['scaler']
encoder = data['encoder']
numeric_cols = data['numeric_cols']
categorical_cols = data['categorical_cols']
encoded_cols = data['encoded_cols']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    
    # Map generalized inputs to the model's expected 3pm features 
    form_data['Humidity3pm'] = form_data.get('Humidity')
    form_data['Pressure3pm'] = form_data.get('Pressure')
    
    input_df = pd.DataFrame([form_data])
    
    # Convert numeric strings to floats [cite: 25]
    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    # Preprocessing pipeline [cite: 429, 436, 683]
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    
    X_input = input_df[numeric_cols + encoded_cols]
    
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1] * 100
    
    result = "Rain is likely tomorrow 🌧️" if prediction == 1 else "No rain expected ☀️"
    return render_template('index.html', 
                           prediction_text=result, 
                           prob_text=f"Confidence: {probability:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)