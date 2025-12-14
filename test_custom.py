from ultralytics import YOLO
import cv2

# 1. Load YOUR custom model (The one you just trained)
model = YOLO('best.pt') 

# 2. Start the Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run detection
    # conf=0.5 means "Only show me if you are 50% sure"
    results = model(frame, conf=0.5) 

    # 4. Draw the results on the screen
    annotated_frame = results[0].plot()

    # 5. Show the video
    cv2.imshow("Disaster Rescue System - Test", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()