

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'ct_scan' not in request.files:
        return jsonify({'error': 'No CT scan uploaded'}), 400

    ct_file = request.files['ct_scan']
    temp_path = 'temp_ct.nii.gz'
    ct_file.save(temp_path)

    # No AI inference, just acknowledge upload
    return jsonify({'message': 'CT scan uploaded successfully. No AI processing performed.'})

if __name__ == '__main__':
    app.run(debug=True)