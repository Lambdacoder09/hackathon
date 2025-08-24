// main.js
// Handles CT scan upload and segmentation preview

const form = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  resultDiv.textContent = '';

  const file = imageInput.files[0];
  if (!file) {
    resultDiv.textContent = 'Please upload a CT scan (.nii or .nii.gz).';
    return;
  }

  // --- Flask backend integration ---
  const formData = new FormData();
  formData.append('ct_scan', file);

  resultDiv.textContent = 'Segmenting...';

  try {
  const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error('Segmentation failed');
    const data = await response.json();
    // { preview: "data:image/png;base64,..." }
    if (data.preview) {
      resultDiv.innerHTML = `<span class="font-bold">Segmentation Preview:</span><br><img src="${data.preview}" alt="Segmentation Preview" class="mx-auto rounded-lg border mt-4" style="max-width:400px;">`;
    } else {
      resultDiv.textContent = 'Segmentation complete, but no preview available.';
    }
  } catch (err) {
    resultDiv.textContent = 'Error: ' + err.message;
  }
});
