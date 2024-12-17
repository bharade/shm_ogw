import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from data.data_loader import DataLoader

class Evaluator:
    def __init__(self, model_path, dataloader):
        self.model_path = model_path
        self.dataloader = dataloader
        self.model = tf.keras.models.load_model(model_path)  # Load your best saved model

    def evaluate(self):
        # Load the test data
        x_test_max, y_test = self.dataloader.load_test_data()

        # Predictions and Pre-processing (if required)
        y_pred_probs = self.model.predict(x_test_max)  # Probabilities
        y_pred = (y_pred_probs > 0.5).astype("int32")  # Convert to binary classes (0 or 1)

        # Plotting the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))  # Adjust size for better visibility
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=["Undamaged", "Damaged"], yticklabels=["Undamaged", "Damaged"])  # Clearer axis labels
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        # Comprehensive Classification Report
        print(classification_report(y_test, y_pred, target_names=["Undamaged", "Damaged"]))

        # Calculating Additional Metrics
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"F1 Score: {f1:.4f}")  # Consistent formatting with 4 decimal places
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    # Instantiate the DataLoader and Evaluator
    dataloader = DataLoader(data_dir="your_data_directory")  # Replace with your data directory
    evaluator = Evaluator(model_path="best_model.keras", dataloader=dataloader)  # Update with your model path
    evaluator.evaluate()