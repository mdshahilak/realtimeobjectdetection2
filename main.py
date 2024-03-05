import datetime
import cv2
import csv
import os
from google.oauth2 import service_account
import gspread

# Set up Google Sheets API credentials
credentials = service_account.Credentials.from_service_account_file('year-project.json', scopes=['https://www.googleapis.com/auth/spreadsheets'])

# Open the Google Sheet
gc = gspread.authorize(credentials)
sheet = gc.open('Your Google Sheet Name').sheet1  # Replace with your sheet name

thres = 0.555

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

csv_file = 'object_detection_data.csv'
image_folder = 'captured_objects'

# Create the image folder if it doesn't exist
os.makedirs(image_folder, exist_ok=True)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Category', 'Date', 'Time', 'Confidence', 'Image Path'])

    while True:
        success, img = cap.read()

        if not success:
            break

        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                timestamp = datetime.datetime.now()

                # Extract date and time components
                date = timestamp.date()
                time = timestamp.strftime("%H:%M:%S")

                if classId > 0 and classId <= len(classNames):
                    class_name = classNames[classId - 1].lower()
                else:
                    class_name = "unknown"

                # Initialize the image path variable
                object_image_path = ""

                # Check if the detected object is a person or an animal
                if class_name == "person" or class_name in ['cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
                    object_category = "person" if class_name == "person" else "animal"

                    # Capture and save the image
                    object_image_path = os.path.join(image_folder, f'{object_category}_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg')
                    cv2.imwrite(object_image_path, img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
                elif class_name in ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']:
                    object_category = "vehicle"
                else:
                    object_category = "object"

                # Check if the object category is not "object" before writing to CSV and Google Sheet
                if object_category != "object":
                    object_data = [object_category, date, time, round(confidence * 100, 2), object_image_path]
                    # Write data to CSV
                    writer.writerow(object_data)

                    # Write data to Google Sheet
                    sheet.append_row(object_data)

                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, object_category.upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                cv2.putText(img, f"Confidence: {round(confidence * 100, 2)}%", (box[0] + 10, box[1] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Object Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
