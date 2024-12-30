import tensorflow as tf
from tf.keras.callbacks import EarlyStopping, ModelCheckpoint
 # Import your DataLoader class
from src.data.data_loader import DataLoader
# Import your Transformer model
from src.models.transformer import TransformerModel
import os
from pathlib import Path
from src.evaluation.evaluator import Evaluator

class Trainer:
    def __init__(self, model, dataloader, epochs, batch_size, learning_rate, patience):
        self.model = model
        self.dataloader = dataloader
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience

        # Define your optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

        # Callbacks for early stopping and model checkpointing
        self.early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
        self.model_checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True)

    def train(self):
        """
        Executes the training loop, incorporating validation and performance tracking.
        """
        # Load your training and validation data
        udam=[]
        dam=[]
        udam,dam = self.dataloader.load_data()
        x_train,x_test,x_val, y_train,y_test,y_val=self.dataloader.split_data(udam,dam)
        print("Data split successfully.")
        
        # Check the new shape
        print(f"x_train_max shape: {x_train.shape}")
        print(f"x_test_max shape: {x_test.shape}")
        
        x_train_max,x_val_max,x_test_max=self.dataloader.downsample_data(x_train,x_val,x_test)
        print("Data downsampled successfully.")

                # Define the path to the artifacts folder relative to training.py
        artifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
        print(f"the shape of each sample is {x_train_max[0].shape}")
        # Ensure the artifacts folder exists
        os.makedirs(artifacts_path, exist_ok=True)
        print(f"Artifacts path: {artifacts_path} created successfully.")
        # Define the checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(artifacts_path, "model_checkpoint_{epoch:02d}.keras"),  # Save in artifacts folder with epoch number
            save_weights_only=False,  # Save the full model
            save_freq='epoch',  # Save at the end of each epoch
            verbose=1  # To print a message when the model is saved
        )
        # Modify the callbacks list to include the checkpoint callback
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True),
            checkpoint_callback
        ]
        # Training with the checkpoint callback
        print("Training started.....:D")
        # Compilation with binary classification loss
        model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                metrics=["accuracy"],
        )
        model.summary()
        
        history = model.fit(
            x_train_max,
            y_train,
            validation_data=(x_val_max, y_val),  # Use x_val_max and y_val for validation
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("Training completed.......:D")
        return x_test_max, y_test
        

if __name__ == "__main__":
    # Instantiate DataLoader and TransformerModel
    dataloader = DataLoader(
        baseline_path=Path("C:/Users/adibh/OneDrive/Desktop/projects/simplified_mtp/shm_ogw/data/Baseline"),
        damage_path=Path("C:/Users/adibh/OneDrive/Desktop/projects/simplified_mtp/shm_ogw/data/Damage")
    )
    #dataloader.load_data()
    
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
    trainer = Trainer(model, dataloader, epochs=200, batch_size=32, learning_rate=1e-4, patience=10)
    x_test_max,y_test=trainer.train()

    # Instantiate the Evaluator
    evaluator = Evaluator(model_path="C:/Users/adibh/OneDrive/Desktop/projects/simplified_mtp/shm_ogw/artifacts/transformer_model.keras")
    evaluator.evaluate(x_test_max=x_test_max, y_test=y_test)