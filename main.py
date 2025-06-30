from ultralytics import YOLO
import cv2

# 1. Load the YOLOv8 model
model = YOLO('yolov8s.pt')  # You can use yolov8n.pt, yolov8m.pt, etc.

# 2. Initialize webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()  # Capture a single frame

if ret:
    # 3. Save the captured image (optional)
    cv2.imwrite("captured.jpg", frame)

    # 4. Run YOLOv8 detection on the captured frame
    results = model(frame)  # or: model("captured.jpg")

    # 5. Plot the results on the image
    annotated_image = results[0].plot()

    # 6. Display the result
    cv2.imshow("YOLOv8 Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Failed to capture image from webcam.")

cap.release()
