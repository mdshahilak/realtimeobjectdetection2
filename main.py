import datetime
import cv2
import os
import pyrebase
import time
import tempfile

firebaseConfig = {
    "apiKey": "AIzaSyC0k6uY_XxRYTsjRqddDqujPf4lKacp2fo",
    "authDomain": "shahil-ai-project.firebaseapp.com",
    "databaseURL": "https://shahil-ai-project-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "shahil-ai-project",
    "storageBucket": "shahil-ai-project.appspot.com",
    "messagingSenderId": "799127360843",
    "appId": "1:799127360843:web:f707a808c47b6d4032c4ce",
}


firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
db = firebase.database()

thres = 0.555

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

image_folder = "captured_objects"
os.makedirs(image_folder, exist_ok=True)


def generate_document_key():
    return "{}".format(int(time.time() * 1000))


while True:
    success, img = cap.read()

    if not success:
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            timestamp = datetime.datetime.now()
            date = timestamp.date()
            time_str = timestamp.strftime("%H:%M:%S")

            if classId > 0 and classId <= len(classNames):
                class_name = classNames[classId - 1].lower()
            else:
                class_name = "unknown"

            object_image_path = ""
            if class_name == "person" or class_name in [
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
            ]:
                object_category = "person" if class_name == "person" else "animal"
                object_image_path = os.path.join(
                    image_folder,
                    f'{object_category}_{timestamp.strftime("%Y%m%d%H%M%S")}.jpg',
                )
                cv2.imwrite(
                    object_image_path,
                    img[box[1] : box[1] + box[3], box[0] : box[0] + box[2]],
                )
            elif class_name in [
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
            ]:
                object_category = "vehicle"
            else:
                object_category = "object"

            if (
                object_category != "object" and confidence > 0.7
            ): 
                with open(object_image_path, "rb") as image_file:
                    temp_local_filename = tempfile.mktemp()
                    with open(temp_local_filename, "wb") as temp_local_file:
                        temp_local_file.write(image_file.read())

                    storage.child("images/" + os.path.basename(object_image_path)).put(
                        temp_local_filename
                    )
                    image_url = storage.child(
                        "images/" + os.path.basename(object_image_path)
                    ).get_url(None)

                # Create a specific document key for each entry
                document_key = generate_document_key()

                # Create a dictionary for the object data
                object_data = {
                    "id": document_key,
                    "category": object_category,
                    "date": str(date),
                    "time": time_str,
                    "confidence": round(float(confidence) * 100, 2),
                    "image_url": image_url,
                }

                # Add data to Firestore with the specific document key
                db.child("object_detection_data").child(document_key).set(object_data)

            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(
                img,
                object_category.upper(),
                (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f"Confidence: {round(float(confidence) * 100, 2)}%",
                (box[0] + 10, box[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Object Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
