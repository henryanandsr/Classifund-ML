from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import AutoModel, AutoTokenizer
import numpy as np
import joblib
import uvicorn
import torch
import gc

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device = torch.device("mps" if mps_available else "cpu")
print(f"Using device: {device}")

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: float

def load_models():
    print("Loading models...")
    try:
        gc.collect()
        if mps_available:
            torch.mps.empty_cache()
        
        bert_model = AutoModel.from_pretrained(
            "bert-base-uncased",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        classifier_model = tf.saved_model.load('final_model')
        print("Model loaded. Available signatures:", classifier_model.signatures.keys())
        classifier_model = classifier_model.signatures['serving_default']
        print("Model input/output details:", classifier_model.structured_outputs)
        
        # Load the label encoder
        label_encoder = joblib.load('label_encoder.joblib')
        
        print("Models loaded successfully")
        return bert_model, tokenizer, classifier_model, label_encoder
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def predict_single_text(text, bert_model, tokenizer, classifier_model, label_encoder):
    try:
        bert_model = bert_model.to(device)
        bert_model.eval()
        
        inputs = tokenizer(
            text, 
            return_tensors="pt",
            truncation=True, 
            max_length=128,
            padding=True
        )
        
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].to('cpu').numpy()
        
        if mps_available:
            torch.mps.empty_cache()
        
        tf_tensor = tf.convert_to_tensor(embedding, dtype=tf.float32)
        predictions = classifier_model(tf_tensor)
        
        output_key = list(predictions.keys())[0]
        prediction_tensor = predictions[output_key]
        prediction_numpy = prediction_tensor.numpy()
        
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction_numpy[0])])[0]
        confidence = float(np.max(prediction_numpy[0]) * 100)
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"Prediction outputs structure: {predictions.keys() if 'predictions' in locals() else 'No predictions made'}")
        raise

try:
    print("Initializing models...")
    bert_model, tokenizer, classifier_model, label_encoder = load_models()
except Exception as e:
    print(f"Failed to load models: {str(e)}")
    raise

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        predicted_class, confidence = predict_single_text(
            request.text,
            bert_model,
            tokenizer,
            classifier_model,
            label_encoder
        )
        
        return PredictionResponse(
            text=request.text,
            predicted_category=predicted_class,
            confidence=confidence
        )
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": all([bert_model, tokenizer, classifier_model, label_encoder])
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        workers=1,
        log_level="info"
    )