
import joblib
import numpy as np
import os

# Load test dataset
X_test = np.load("data/processed/X_test.npy")

# Model folder
model_dir = "results/models"

print("\nTesting saved models...\n")

for filename in os.listdir(model_dir):

    if filename.endswith(".joblib"):

        model_path = os.path.join(model_dir, filename)

        # Load saved model
        model = joblib.load(model_path)

        # Generate predictions
        predictions = model.predict(X_test)

        # Verify predictions
        assert len(predictions) == len(X_test)
        assert np.isfinite(predictions).all()

        print(f"{filename}: SUCCESS")
        print(f"Predictions shape: {predictions.shape}")
        print(f"First 3 predictions: {predictions[:3]}\n")

print("All saved models loaded and tested successfully!")
