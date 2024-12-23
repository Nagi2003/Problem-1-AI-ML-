from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)

with open('xgb_model1.pkl', 'rb') as file:
    model = pickle.load(file)

one_hot_cols = ['Origin', 'Destination', 'Vehicle Type', 'Weather Conditions']

one_hot_categories = {
    'Origin': ['Jaipur', 'Bangalore', 'Mumbai', 'Hyderabad', 'Chennai', 'Kolkata', 'Lucknow', 'Delhi', 'Ahmedabad', 'Pune'],
    'Destination': ['Mumbai', 'Delhi', 'Chennai', 'Ahmedabad', 'Kolkata', 'Lucknow', 'Bangalore', 'Pune', 'Jaipur', 'Hyderabad'],
    'Vehicle Type': ['Trailer', 'Truck', 'Container', 'Lorry'],
    'Weather Conditions': ['Rain', 'Storm', 'Clear', 'Fog']
}


traffic_mapping = ['Light', 'Moderate', 'Heavy']

# Initialize and fit the OrdinalEncoder on traffic_mapping
ordinal_encoder = OrdinalEncoder(categories=[traffic_mapping])
ordinal_encoder.fit(pd.DataFrame({'Traffic Conditions': traffic_mapping}))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        origin = request.form['origin']
        destination = request.form['destination']
        vehicle_type = request.form['vehicle_type']
        weather_conditions = request.form['weather_conditions']
        traffic_conditions = request.form['traffic_conditions']
        distance = int(request.form['distance'])

        # Create DataFrame for input data
        input_data = pd.DataFrame({
            'Origin': [origin],
            'Destination': [destination],
            'Vehicle Type': [vehicle_type],
            'Weather Conditions': [weather_conditions]
        })
        
        input_data_encoded = pd.get_dummies(input_data, columns=one_hot_cols)

        for feature, categories in one_hot_categories.items():
            for category in categories:
                col_name = f"{feature}_{category}"
                if col_name not in input_data_encoded.columns:
                    input_data_encoded[col_name] = 0


        traffic_encoded = ordinal_encoder.transform([[traffic_conditions]])

        input_data_encoded['Traffic Conditions'] = traffic_encoded.flatten()

        input_data_encoded['Distance'] = distance

        features = input_data_encoded.values

        prediction = model.predict(features)

        result = 'Delayed' if prediction[0] == 1 else 'Not Delayed'

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
