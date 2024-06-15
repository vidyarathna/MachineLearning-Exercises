from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('path_to_your_saved_model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['input'])  # Assuming input is a list of features
    input_data = input_data.reshape(1, -1)  # Reshape as needed
    
    prediction = model.predict(input_data)
    output = {'prediction': float(prediction[0][0])}  # Assuming a single output
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
