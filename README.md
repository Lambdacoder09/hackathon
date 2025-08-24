# Liver CT Segmentation Viewer

A web application for segmenting liver anomalies from CT scans using AI. Upload a CT scan to view segmentation results and preview the output.

## Features
- Upload CT scans in `.nii` or `.nii.gz` format
- AI-based segmentation (UNet, PyTorch Lightning, TorchIO)
- Segmentation preview in browser
- Gradio interface for interactive exploration
- Modern UI with Tailwind CSS

## Project Structure
```
app.py           # Flask backend (API endpoint for segmentation)
index.html       # Frontend UI
main.js          # Handles upload and preview logic
model.py         # AI model, training, and Gradio app
```

## Setup Instructions

### 1. Python Environment
- Python 3.8+
- Recommended: Create a virtual environment

### 2. Install Dependencies
```bash
pip install flask flask-cors nibabel gradio matplotlib numpy celluloid torch torchio pytorch-lightning
```

### 3. Run the Backend
```bash
python app.py
```
- The Flask server will start at `http://127.0.0.1:5000`

### 4. Open the Frontend
- Open `index.html` in your browser
- Upload a CT scan and click "Segment & Preview"

### 5. Gradio Demo (Optional)
```bash
python model.py
```
- Launches an interactive Gradio app for segmentation

## API
### POST `/predict`
- **Request:** Multipart form with `ct_scan` file
- **Response:**
  - `{ "preview": "data:image/png;base64,..." }` (if implemented)
  - `{ "message": "CT scan uploaded successfully. No AI processing performed." }` (default)

## Team
- Zayed
- Gaayatri


## License
MIT
