import datetime
import cv2

# Set the threshold to detect objects
thres = 0.45

# Initialize the video capture from your webcam (camera index 0)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
cap.set(10, 70)   # Set brightness

# Load class names from a file
classNames = []
classFile = 'coco.names'  # Replace with the actual path to your class names file
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Replace with your model config path
weightsPath = 'frozen_inference_graph.pb'  # Replace with your model weights path

# Create the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Check if the frame is empty (i.e., not captured successfully)
    if not success:
        break

    # Detect objects in the frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Get the current timestamp
            timestamp = datetime.datetime.now()

            # Check the class name (add error handling to avoid IndexError)
            if classId > 0 and classId <= len(classNames):
                class_name = classNames[classId - 1].lower()
            else:
                class_name = "unknown"

            # Determine whether to display the object name as "person" or "object"
            if "person" in class_name or "vehicle" in class_name:
                object_name = class_name
            else:
                object_name = "object"

            # Create a string with the object name and timestamp
            object_data = f"{object_name} at {timestamp}\n"

            # Append the data to a text file
            with open('object_detection_data.txt', 'a') as text_file:
                text_file.write(object_data)

            # Draw a rectangle around the detected object
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

            # Display the class name and confidence
            cv2.putText(img, object_name.upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Object Detection", img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
