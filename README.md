**MRI-Based Brain Tumor Classification**

Deep learning project for brain tumor detection and classification using EfficientNet-B0 and transfer learning.

**Description**:
- This repository provides a FastAPI backend that serves a trained TensorFlow EfficientNet-B0 model to detect and classify brain tumors from MRI scans. The project includes image preprocessing, model weights, and a prediction API.

**Key Features**:
- FastAPI backend: lightweight API for predictions.
- TensorFlow: model training and inference.
- EfficientNet-B0: transfer learning for high accuracy.
- MRI image preprocessing: resizing, normalization, and augmentation utilities.
- Prediction API: HTTP endpoint to upload MRI images and receive tumor classification.

**Project Structure**:
- backend/: FastAPI app and inference code.
- backend/models/: Trained model weights (e.g., `efficientnet_b0_best.h5`).
- backend/requirements.txt: Python dependencies for the backend.
- README.md: This file.

**Getting Started (Backend)**
1. Create a Python environment (recommended):

```
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate
```

2. Install dependencies:

```
pip install -r backend/requirements.txt
```

3. Run the FastAPI server (from the `backend/` folder):

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Model**
- The trained model is located at `backend/models/efficientnet_b0_best.h5` and is loaded by the backend for inference. Replace or update this file if you retrain the model.

**Prediction API (example)**
- Endpoint: `POST /predict` (multipart/form-data with an image file) — returns JSON with predicted class and confidence score.

**Notes**
- Ensure image inputs are MRI slices (grayscale or 3-channel converted) and follow the preprocessing pipeline used during training (resize to model input, normalize pixel values).
- If you retrain the model, update `backend/models/` and optionally log training details in `backend/`.

**Contributing**
- Open issues or pull requests for improvements, bug fixes, or model updates.

**License**
- Refer to repository owner for license details.
