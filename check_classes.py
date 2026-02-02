from ultralytics import YOLO

# Load your model
model = YOLO("best.pt")

# Print the classes it knows
print("🧠 My Model knows these classes:")
print(model.names)