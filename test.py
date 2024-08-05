# test.py

import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Function to count people in a frame
def count_people(frame):
    # Perform object detection
    results = model(frame)

    # Filter results to include only 'person' class
    person_class_id = 0  # COCO class ID for 'person'
    person_detections = [d for d in results[0].boxes.data if d[5] == person_class_id]

    # Count number of people detected
    num_people = len(person_detections)

    # Draw bounding boxes around detected people
    for detection in person_detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Return the count of people detected and the frame with detections
    return num_people, frame

if __name__ == '__main__':
    # Open the device camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        exit()
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Count people in the frame
        num_people, frame_with_detections = count_people(frame)

        # Display the count of people on the frame
        cv2.putText(frame_with_detections, f'People Count: {num_people}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the frame with detections
        cv2.imshow('People Counter', frame_with_detections)

        # Print number of people detected
        print(f'Number of people detected: {num_people}')

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
