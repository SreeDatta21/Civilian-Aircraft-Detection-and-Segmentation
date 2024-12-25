from ultralytics import YOLO

# Load the model from your local path
model = YOLO("Path to your model (Detection.pt or Segmentation.pt)")

# Perform prediction on your local image and save the results
results = model.predict(
    source='Path of Image',
    save=True,
    save_dir='Path to store results'
)
