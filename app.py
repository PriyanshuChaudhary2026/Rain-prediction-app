from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
def load_model():
    try:
        with open('rain_model.pkl', 'rb') as f:
            data = pickle.load(f)
            return data['model'], data['scaler']
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'})
    
    try:
        # Get data from form
        data = {
            'pressure': float(request.form['pressure']),
            'temparature': float(request.form['temperature']),
            'dewpoint': float(request.form['dewpoint']),
            'humidity': float(request.form['humidity']),
            'cloud': float(request.form['cloud']),
            'sunshine': float(request.form['sunshine']),
            'winddirection': float(request.form['winddirection']),
            'windspeed': float(request.form['windspeed'])
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([list(data.values())], 
                              columns=['pressure', 'temparature', 'dewpoint', 'humidity',
                                      'cloud', 'sunshine', 'winddirection', 'windspeed'])
        
        # Scale and predict
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        result = {
            'prediction': 'Rain' if prediction == 1 else 'No Rain',
            'probability': f"{probability:.2%}",
            'input_data': data
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/random')
def random_prediction():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'})
    
    try:
        # Generate random but realistic weather data
        data = {
            'pressure': round(np.random.uniform(980, 1040), 2),
            'temperature': round(np.random.uniform(-10, 40), 2),
            'dewpoint': round(np.random.uniform(-15, 25), 2),
            'humidity': round(np.random.uniform(10, 100), 2),
            'cloud': round(np.random.uniform(0, 100), 2),
            'sunshine': round(np.random.uniform(0, 14), 2),
            'winddirection': round(np.random.uniform(0, 360), 2),
            'windspeed': round(np.random.uniform(0, 100), 2)
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([list(data.values())], 
                              columns=['pressure', 'temparature', 'dewpoint', 'humidity',
                                      'cloud', 'sunshine', 'winddirection', 'windspeed'])
        
        # Scale and predict
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        result = {
            'prediction': 'Rain' if prediction == 1 else 'No Rain',
            'probability': f"{probability:.2%}",
            'input_data': data
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)