// main.js
// Handles image upload and risk prediction

const form = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  resultDiv.textContent = '';

  const file = imageInput.files[0];
  if (!file) {
    resultDiv.textContent = 'Please upload an image.';
    return;
  }

  // --- Flask backend integration ---
  const formData = new FormData();
  formData.append('image', file);

  resultDiv.textContent = 'Checking risk...';

  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error('Prediction failed');
    const data = await response.json();
    // { risk: "Low", probability: 92.3 }
    resultDiv.innerHTML = `<span class="font-bold">${data.risk} Risk</span><br><span class="text-gray-600">Probability: ${data.probability}%</span>`;
  } catch (err) {
    resultDiv.textContent = 'Error: ' + err.message;
  }
});

// --- If using TF.js, replace above with model loading and prediction code ---
// Example:
// import * as tf from '@tensorflow/tfjs';
// Load model, preprocess image, run prediction, display result
