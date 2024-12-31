import tensorflow as tf
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
        self.early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
        self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)

    def get_data(self):
        # Load your training and validation data
        udam=[]
        dam=[]
        udam,dam = self.dataloader.load_data()
        self.x_train,self.x_test,self.x_val, self.y_train,self.y_test,self.y_val=self.dataloader.split_data(udam,dam)
        print("Data split successfully.")
        
        # Check the new shape
        print(f"x_train_max shape: {self.x_train.shape}")
        print(f"x_test_max shape: {self.x_test.shape}")
        self.x_train_max,self.x_val_max,self.x_test_max=self.dataloader.downsample_data(self.x_train,self.x_val,self.x_test)
        print("Data downsampled successfully.")

    def train(self):
        """
        gets the data and trains the model
        Executes the training loop, incorporating validation and performance tracking.
        """
        self.get_data()
        artifacts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
        print(f"the shape of each sample is {self.x_train_max[0].shape}")
        # Ensure the artifacts folder exists
        os.makedirs(artifacts_path, exist_ok=True)
        print(f"Artifacts path: {artifacts_path} created successfully.")
        # Define the checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(artifacts_path, "transformer_model_new.keras"),  # Save in artifacts folder with epoch number
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
        self.model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                metrics=["accuracy"],
        )
        self.model.summary()
        
        history = self.model.fit(
            self.x_train_max,
            self.y_train,
            validation_data=(self.x_val_max, self.y_val),  # Use x_val_max and y_val for validation
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("Training completed.......:D")
        self.evaluate()
    
    def evaluate(self):
        # Instantiate the Evaluator
        evaluator = Evaluator(model_path="C:/Users/adibh/OneDrive/Desktop/projects/simplified_mtp/shm_ogw/artifacts/transformer_model.keras")
        evaluator.evaluate(x_test_max=self.x_test_max, y_test=self.y_test)
        


