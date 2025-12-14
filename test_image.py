from ultralytics import YOLO
import cv2

# 1. Load your custom brain
model = YOLO('best.pt')

# 2. Run detection on the specific image file
# conf=0.4 means "Only detect if 40% sure"
results = model('test.jpg', conf=0.1)

# 3. Show the results
for result in results:
    # Draw the boxes
    annotated_frame = result.plot()

    # Show the image in a window
    cv2.imshow("Detection Result", annotated_frame)

    # Save the output so you can send it to your friends
    cv2.imwrite("result_output.jpg", annotated_frame)
    print("✅ Saved result to 'result_output.jpg'")

# 4. Wait until you press any key to close
cv2.waitKey(0)
cv2.destroyAllWindows()