from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import tempfile
from PIL import Image
import os

app = FastAPI()

# CORS for development (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mri-brain-tumor-classification-effi.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResult(BaseModel):
    tumor_detected: bool
    confidence_score: float
    tumor_type: str | None
    tumor_location: str | None
    analysis_notes: str

# Load your trained model
model = load_model("models/efficientnet_b0_best.h5", compile=False)


# Class names in correct order
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.post("/predict", response_model=AnalysisResult)
async def predict(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    doctor_id: str = Form(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:

        img = Image.open(tmp_path).convert("RGB")
        img = img.resize((224, 224))

        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x, verbose=0)[0]   # shape: (4,)

        # Top prediction
        pred_index = int(np.argmax(preds))
        confidence_score = float(preds[pred_index] * 100)
        raw_class = class_names[pred_index]

        # Tumor presence
        tumor_detected = raw_class != "notumor"

        DISPLAY_NAMES = {
            "glioma": "Glioma",
            "meningioma": "Meningioma",
            "pituitary": "Pituitary Tumor",
            "notumor": "No Tumor"
        }

        tumor_type = DISPLAY_NAMES[raw_class]

        # Confidence-aware medical notes
        if tumor_detected:
            if confidence_score >= 85:
                analysis_notes = (
                    f"{tumor_type} detected with high confidence. "
                    "Findings strongly suggest pathology. Clinical correlation advised."
                )
            elif confidence_score >= 65:
                analysis_notes = (
                    f"{tumor_type} detected with moderate confidence. "
                    "Further imaging or expert review recommended."
                )
            else:
                analysis_notes = (
                    f"Possible {tumor_type.lower()} detected with low confidence. "
                    "Manual radiologist review is required."
                )
        else:
            analysis_notes = (
                "No tumor detected. Brain MRI appears within normal limits. "
                "If symptoms persist, further evaluation is recommended."
            )

        return AnalysisResult(
            tumor_detected=tumor_detected,
            confidence_score=round(confidence_score, 2),
            tumor_type=tumor_type if tumor_detected else None,
            tumor_location="Frontal Lobe" if tumor_detected else None,  # placeholder
            analysis_notes=analysis_notes,
        )

    finally:
        os.remove(tmp_path)
