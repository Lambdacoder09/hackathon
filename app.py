
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import nibabel as nib
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

# --- Load your trained segmentation model checkpoint ---
# Place your model checkpoint (e.g., 'segmenter.ckpt') in the project folder
MODEL_PATH = 'segmenter.ckpt'  # Update with your actual checkpoint file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define your UNet and Segmenter classes here or import them ---
from model import UNet, Segmenter  # Assumes model.py contains these classes

model = Segmenter.load_from_checkpoint(MODEL_PATH)
model = model.eval()
model = model.to(device)

app = Flask(__name__)
CORS(app)

def get_segmentation_preview(ct_path):
    img = nib.load(ct_path).get_fdata()
    img = (img - img.min()) / (img.max() - img.min())
    volume_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(volume_tensor)
        mask = torch.argmax(pred, dim=1).cpu().squeeze().numpy()
    # Get mid-slice
    z = img.shape[2] // 2
    slice2d, mask2d = img[:, :, z], mask[:, :, z]
    fig, ax = plt.subplots()
    ax.imshow(slice2d, cmap="gray")
    ax.imshow(mask2d, alpha=0.4, cmap="jet")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.getvalue()
    base64_img = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{base64_img}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'ct_scan' not in request.files:
        return jsonify({'error': 'No CT scan uploaded'}), 400

    ct_file = request.files['ct_scan']
    # Save uploaded file to a temporary location
    temp_path = 'temp_ct.nii.gz'
    ct_file.save(temp_path)

    preview = get_segmentation_preview(temp_path)

    return jsonify({'preview': preview})

if __name__ == '__main__':
    app.run(debug=True)