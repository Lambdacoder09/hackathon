from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)  # Allows frontend requests

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # image = request.files['image']
    # Here you would process the image and run your model

    # Dummy prediction logic
    risks = ['Low', 'Medium', 'High']
    risk = random.choice(risks)
    probability = round(random.uniform(60, 100), 1)

    return jsonify({'risk': risk, 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)