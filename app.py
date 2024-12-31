from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
import uvicorn
import os
from src.prediction.predictor import Predictor
from src.models.transformer import TransformerModel
from src.training.trainer import Trainer
from src.data.data_loader import DataLoader
from main import train_model, predict_sample  # Import functions from main.py
app = FastAPI()

@app.get("/", tags=["root"])
async def index():
    """
    Redirect to API docs for convenience.
    """
    return RedirectResponse(url="/docs")
@app.get("/train")
async def training_route():
    """
    Trigger the training process.
    """
    try:
        result = train_model()  # Call the train_model function
        return {"message": "Training Successful", "details": result}
    except Exception as e:
        return Response(f"Error occurred during training: {e}")
@app.post("/predict")
async def predict_route(model_path: str, sample_path: str):
    """
    Predict the output for a given sample using the pre-trained model.
    """
    try:
        prediction = predict_sample(model_path, sample_path)
        formatted_predictions = [
            {"Sample": i + 1, "Status": "Damaged" if pred == 1 else "Undamaged"}
            for i, pred in enumerate(prediction)
        ]
        return {"message": "Prediction Successful", "predictions": formatted_predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
