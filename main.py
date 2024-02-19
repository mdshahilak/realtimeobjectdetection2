import datetime
import cv2

thres = 0.45

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

while True:
    success, img = cap.read()

    if not success:
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            timestamp = datetime.datetime.now()

            if classId > 0 and classId <= len(classNames):
                class_name = classNames[classId - 1].lower()
            else:
                class_name = "unknown"

            object_name = class_name if "person" in class_name or "vehicle" in class_name else "object"
            object_data = f"{object_name} at {timestamp}\n"

            with open('object_detection_data.txt', 'a') as text_file:
                text_file.write(object_data)

            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, object_name.upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Object Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
