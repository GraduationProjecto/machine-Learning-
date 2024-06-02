# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 14:45:08 2024

@author: rowyd
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import joblib

# Load and clean the dataset
car = pd.read_csv('quikr_car.csv')

# Data cleaning
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != "Ask For Price"]
car['Price'] = car['Price'].str.replace(',', '').astype(int)
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split(' ').str.slice(0, 3).str.join(' ')
car = car.reset_index(drop=True)
car = car[car['Price'] < 6e6].reset_index(drop=True)
car.to_csv('cleaned_car.csv')

# Model training
x = car.drop(columns='Price')
y = car['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(x[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)
print(f"Model R2 score: {r2_score(y_test, y_pred)}")

scores = []
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    scores.append(r2_score(y_test, y_pred))

best_score = np.argmax(scores)
print(f"Best score: {scores[best_score]} at random state {best_score}")

# Save the model
joblib.dump(pipe, './LinearRegressionModel.pkl')

# Load the model pipeline
model = joblib.load('./LinearRegressionModel.pkl')

# Define the Flask application
app = Flask(__name__)
cors = CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # Convert the data into a DataFrame
        input_data = pd.DataFrame([data])
        
        # Ensure the data types are correct
        input_data['year'] = input_data['year'].astype(int)
        input_data['kms_driven'] = input_data['kms_driven'].astype(int)
        
        # Make a prediction using the loaded model pipeline
        prediction = model.predict(input_data)
        
        # Convert the prediction to an integer
        prediction_int = int(prediction[0]* 2)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction_int })
    except Exception as e:
        # Return any errors as a JSON response
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app with host 0.0.0.0 and port 8080
    app.run(host='0.0.0.0', port=8080)

