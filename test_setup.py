from ultralytics import YOLO
import cv2

# Load the official tiny model
print("Downloading model...")
model = YOLO('yolo11n.pt') 

# Run a test on a random image from the internet
print("Running detection...")
results = model('https://ultralytics.com/images/bus.jpg', show=True)

# Keep window open
cv2.waitKey(0)
cv2.destroyAllWindows()