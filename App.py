from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import io

app = FastAPI()

# Configuración de YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_objects(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(
        image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    return class_ids, confidences, boxes


def is_wearing_glasses(class_ids, classes):
    for class_id in class_ids:
        if classes[class_id] == 'glasses':
            return True
    return False


def is_wearing_cap(class_ids, classes):
    for class_id in class_ids:
        if classes[class_id] == 'cap':
            return True
    return False


@app.post("/detect_objects/")
async def detect_objects_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)

    class_ids, confidences, boxes = detect_objects(image)

    if is_wearing_glasses(class_ids, classes) and is_wearing_cap(class_ids, classes):
        return {"result": "Person is wearing glasses and a cap."}
    elif is_wearing_glasses(class_ids, classes):
        return {"result": "Person is wearing glasses."}
    elif is_wearing_cap(class_ids, classes):
        return {"result": "Person is wearing a cap."}
    else:
        return {"result": "Person is not wearing glasses or a cap."}