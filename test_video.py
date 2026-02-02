import cv2
from ultralytics import YOLO

# --- CONFIGURATION (CHANGE THESE IF NEEDED) ---
MODEL_PATH = "best.pt"              # Your trained model
VIDEO_PATH = "test_video.mp4"       # PUT YOUR VIDEO FILE NAME HERE!
OUTPUT_PATH = "output_video.mp4"    # The name of the result file

# --- 1. LOAD THE MODEL ---
print(f"🚀 Loading model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- 2. OPEN THE VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Error: Could not find video file '{VIDEO_PATH}'")
    print("👉 Make sure you dragged a video into this folder and renamed it!")
    exit()

# Get video properties to save it correctly
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Setup the Video Writer (to save the result)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("🎬 Starting detection... Press 'q' to stop early.")

# --- 3. RUN DETECTION LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # Run YOLO inference on the frame
        # persist=True helps the model 'remember' objects between frames (smoother boxes)
        results = model.track(frame)

        # Draw the boxes on the frame
        annotated_frame = results[0].plot()

        # Display the frame on your screen
        cv2.imshow("YOLO Disaster Detection (Press 'q' to Quit)", annotated_frame)

        # Save the frame to the output video
        out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Video ended
        break

# --- 4. CLEANUP ---
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Finished! Your video is saved as: {OUTPUT_PATH}")