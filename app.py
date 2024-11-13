from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler (Assuming Random Forest model for this example)
with open(r'C:/Users/HP/Desktop/AI3163/AI3163/random_forest_model.pkl', 'rb') as f:
    # Load your model here
    model = pickle.load(f)

with open('C:/Users/HP/Desktop/AI3163/AI3163/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather form data with default values for safety
        data = [
            request.form.get('age', type=float, default=0),
            request.form.get('gender', type=int, default=0),  # Assuming gender is coded as an integer
            request.form.get('university_year', type=int, default=1),
            request.form.get('sleep_duration', type=float, default=0),
            request.form.get('study_hours', type=float, default=0),
            request.form.get('screen_time', type=float, default=0),
            request.form.get('caffeine_intake', type=float, default=0),
            request.form.get('physical_activity', type=float, default=0),
            request.form.get('weekday_sleep_start', type=float, default=0),
            request.form.get('weekend_sleep_start', type=float, default=0),
            request.form.get('weekday_sleep_end', type=float, default=0),
            request.form.get('weekend_sleep_end', type=float, default=0)
        ]

        # Ensure all values were gathered
        if None in data:
            return jsonify({'error': 'Some form fields are missing or invalid.'}), 400

        # Convert data to array, reshape, and scale
        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)  # Scaling the input features

        # Predict sleep quality
        prediction =  "Good" if  model.predict(data)[0] >=8 else "poor"  
        print(prediction)
        
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
