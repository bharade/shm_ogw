import os 
import sys
from pathlib import Path
from src.data.data_loader import DataLoader
from src.models.transformer import TransformerModel
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.prediction.predictor import Predictor
def train_model(epochs=200, batch_size=32, learning_rate=1e-4, patience=10):
    """
    Train the Transformer model.
    """
    dataloader = DataLoader(
        baseline_path=Path("C:/Users/adibh/OneDrive/Desktop/projects/simplified_mtp/shm_ogw/data/Baseline"),
        damage_path=Path("C:/Users/adibh/OneDrive/Desktop/projects/simplified_mtp/shm_ogw/data/Damage"),
    )
    
    model = TransformerModel(
        input_shape=(874,), 
        head_size=512,
        num_heads=8,
        ff_dim=128,
        num_transformer_blocks=4,
        mlp_units=[256],
        mlp_dropout=0.4,
        dropout=0.3,
    )
    print("Model created successfully.")
    trainer = Trainer(model, dataloader, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
    history=trainer.train()
    return history

def predict_sample(model_path, sample_path):
    """
    Predict using a pre-trained Transformer model.
    """
    # model_path = "C:/Users/adibh/OneDrive/Desktop/projects/simplified_mtp/shm_ogw/artifacts/transformer_model.keras"
    # path_to_sample = "C:/Users/adibh/OneDrive/Desktop/projects/simplified_mtp/shm_ogw/data"
    predictor = Predictor(model_path, sample_path)
    predictor.process_sample()
    predictor.normlize_data()
    x_pred_max = predictor.downsample_data()
    prediction = predictor.predict(x_pred_max)
    return prediction