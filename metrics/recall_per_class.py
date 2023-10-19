import tensorflow as tf
from sklearn.metrics import recall_score
import numpy as np

from classification.training.train import SkinLesionCNNTrainer, get_training_settings


trainer = SkinLesionCNNTrainer(**get_training_settings(), n_classes=5)


# Load your trained model and test data
experiment_folder = "mix_40_original_60_generated"
model = tf.keras.models.load_model(
    f"models/classification/{experiment_folder}/cnn_model_original.h5"
)

# Test data
test_data = trainer._get_test_dataloader(folder=experiment_folder, shuffle=False)

# Make predictions on the test data
y_pred = model.predict(test_data)

# Convert predicted probabilities to class labels
predicted_labels = np.argmax(y_pred, axis=1)

# Get the true class labels from the generator
true_labels = test_data.classes

# Calculate recall per class
recall_per_class = recall_score(true_labels, predicted_labels, average=None)

# Print or use the recall values as needed
print("Recall per class:", recall_per_class)
