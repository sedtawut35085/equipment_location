from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os
import sys

# Add parent directory to path to import predict_util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_util import load_prediction_model, predict_single_image, MODEL_PATH, CLASS_NAMES

app = FastAPI(
    title="Equipment Classifier API",
    description="API for classifying equipment images",
    version="1.0.0"
)

class_names = CLASS_NAMES  # Use class names from environment config

# Load model with error handling using predict_util
try:
    # Use the MODEL_PATH from predict_util, but adjust for API directory
    api_model_path = os.path.join("..", MODEL_PATH)
    model = load_prediction_model(api_model_path)
    if model is None:
        raise Exception("Model loading failed")
    print(f"✅ Model loaded successfully using predict_util")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.get("/")
async def root():
    return {"message": "Meat Classifier API is running! Use /predict endpoint to classify meat images."}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "message": "Model loaded successfully" if model is not None else "Model not loaded"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))  # ปรับตาม input shape ของโมเดลคุณ
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # ถ้าเทรนด้วย normalize

        prediction = model.predict(img_array)
        result = prediction.tolist()  # แปลงให้อ่านง่าย
        
        # Add interpretation of results
    
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {
            "prediction": result,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "filename": file.filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
