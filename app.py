from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names used during training (from your X_train.columns)
training_features = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
    'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

# Encoding dictionaries (based on your LabelEncoder from the notebook)
job_map = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11}
marital_map = {'divorced': 0, 'married': 1, 'single': 2, 'unknown': 3}
education_map = {'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3, 'illiterate': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': 7}
housing_map = {'no': 0, 'yes': 1, 'unknown': 2}
loan_map = {'no': 0, 'yes': 1, 'unknown': 2}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
        
        
        age = int(data['age'])
        job = data['job']
        marital = data['marital']
        education = data['education']
        housing = data['housing']
        loan = data['loan']
        campaign = int(data['campaign'])

        # Encode categorical variables
        job_encoded = job_map.get(job, 11)  # Default to 'unknown' if not found
        marital_encoded = marital_map.get(marital, 3)
        education_encoded = education_map.get(education, 7)
        housing_encoded = housing_map.get(housing, 2)
        loan_encoded = loan_map.get(loan, 2)

        # Set default values for other features 
        input_data = {
            'age': [age],
            'job': [job_encoded],
            'marital': [marital_encoded],
            'education': [education_encoded],
            'default': [0],  # Assuming 'no'
            'housing': [housing_encoded],
            'loan': [loan_encoded],
            'contact': [0],  # Assuming 'telephone'
            'month': [5],  # Assuming 'may'
            'day_of_week': [0],  # Assuming 'mon'
            'duration': [700],  # Fixed value
            'campaign': [campaign],
            'pdays': [999],  # Fixed value
            'previous': [0],  # Fixed value
            'poutcome': [0],  # Assuming 'nonexistent'
            'emp.var.rate': [1.4],  # Fixed value
            'cons.price.idx': [93.918],  # Fixed value
            'cons.conf.idx': [-42.7],  # Fixed value
            'euribor3m': [4.961],  # Fixed value
            'nr.employed': [5228.1]  # Fixed value
        }

        # Create DataFrame and ensure correct feature order
        df_input = pd.DataFrame(input_data)
        df_input = df_input[training_features]

        # Scale the input
        df_input_scaled = scaler.transform(df_input)

        # Make prediction
        prediction = svm_model.predict(df_input_scaled)
        result = 'yes' if prediction[0] == 1 else 'no'

        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)