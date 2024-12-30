import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
 # Import your DataLoader class
from src.data.data_loader import DataLoader
# Import your Transformer model
from src.models.transformer import TransformerModel
import os
from pathlib import Path


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
        history = model.fit(
            x_train_max,
            y_train,
            validation_data=(x_val_max, y_val),  # Use x_val_max and y_val for validation
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        print("Training completed.......:D")

        # # Training loop
        # for epoch in range(self.epochs):
        #     print(f"Epoch {epoch + 1}/{self.epochs}")

        #     # Train on batches of data
        #     for batch in range(len(x_train_max) // self.batch_size):
        #         with tf.GradientTape() as tape:
        #             start = batch * self.batch_size
        #             end = (batch + 1) * self.batch_size
        #             logits = self.model(x_train_max[start:end])
        #             loss = self.loss_fn(y_train[start:end], logits)
                
        #         # Compute gradients and update model weights
        #         grads = tape.gradient(loss, self.model.trainable_variables)
        #         self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        #     # Validation at the end of each epoch
        #     val_logits = self.model(x_val_max)
        #     val_loss = self.loss_fn(y_val, val_logits)
        #     print(f"Validation Loss: {val_loss:.4f}")

        #     # Early stopping and model checkpointing
        #     self.early_stopping(val_loss=val_loss)
        #     self.model_checkpoint(epoch=epoch, logs={'val_loss': val_loss})

        #     if self.early_stopping.stopped_epoch > 0:
        #         print(f"Early stopping at epoch {self.early_stopping.stopped_epoch}")
        #         break
            
    def evaluate(self):
        """
        Evaluates the trained model on the test set using relevant metrics.
        """
        x_test_max, y_test = self.dataloader.load_test_data()
        test_logits = self.model(x_test_max)
        test_loss = self.loss_fn(y_test, test_logits).numpy()
        test_accuracy = tf.keras.metrics.binary_accuracy(y_test, test_logits).numpy()
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

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

    # Instantiate Trainer and start training
    trainer = Trainer(model, dataloader, epochs=200, batch_size=32, learning_rate=1e-4, patience=10)
    trainer.train()

    # Evaluate the best model on the test set
    trainer.evaluate()