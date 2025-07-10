from ultralytics import YOLO
import os

# 1. SETUP YOUR FILE PATHS
DATA_YAML_PATH = "data.yaml"  # Path to your data.yaml file
MODEL_TYPE = "yolov8n.yaml"   # Choose yolov8n, yolov8s, etc.
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16

# 2. START TRAINING
def train_model():
    print("[INFO] Loading model...")
    model = YOLO(MODEL_TYPE)

    print("[INFO] Starting training...")
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH
    )
    print("[DONE] Training complete. Model saved to: runs/detect/train/weights/best.pt")

# 3. OPTIONAL: RUN DETECTION ON AN IMAGE
def run_inference(image_path):
    model = YOLO("runs/detect/train/weights/best.pt")  # Load best model
    results = model(image_path, save=True)
    print("[INFO] Detection complete. Image saved in 'runs/detect/predict/'.")

# -------------------- USAGE --------------------

if __name__ == "__main__":
    train_model()

    # Uncomment to test a detection after training
    # run_inference("test_image.jpg")
