import uvicorn
import shutil
import time
from fastapi.responses import FileResponse, StreamingResponse
from datetime import datetime
import csv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import io
import os
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def detect_objects(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(
        image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    human_count = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':
                human_count += 1

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    return class_ids, confidences, boxes, human_count


def eliminar_simbolos_timestamp(timestamp):
    fecha_hora = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    fecha_hora_formateada = fecha_hora.strftime("%Y%m%d%H%M%S")

    return fecha_hora_formateada


@app.post("/predict/")
async def detect_objects_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)

    start_time = time.time()

    class_ids, confidences, boxes, human_count = detect_objects(image)

    end_time = time.time()
    execution_time = end_time - start_time

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    current_directory = os.path.dirname(os.path.abspath(__file__))

    save_folder = os.path.join(current_directory, "humans_detected")
    print("Ruta completa para la carpeta de destino:", save_folder)

    os.makedirs(save_folder, exist_ok=True)
    print("¿La carpeta existe ahora?", os.path.exists(save_folder))

    save_path = os.path.join(
        save_folder, f"detected_{eliminar_simbolos_timestamp(timestamp)}.jpg")
    print("Ruta completa para guardar la imagen:", save_path)

    cv2.imwrite(save_path, image)

    report_info = [file.filename, f"{image.shape[1]}x{image.shape[0]}",
                   human_count, timestamp, execution_time, "YOLOv3", save_path]
    prediction_reports.append(report_info)

    response_data = {
        "timestamp": timestamp,
        "human_count": human_count,
        "question": "Eres tu?" if human_count > 0 else "Todo tranquilo de momento",
        "download_link": save_path  # Agregar un enlace de descarga al JSON
    }

    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = io.BytesIO(img_encoded.tobytes())

    return StreamingResponse(img_bytes, media_type="image/png", headers={"X-Info": str(response_data)})

@app.get("/status")
async def get_status():
    service_info = {"status": "running",
                    "message": "Service is up and running."}

    # Información del modelo
    model_info = {
        "model_name": "YOLOv3",
        "weights_file": "yolov3.weights",
        "config_file": "yolov3.cfg",
        "classes_file": "coco.names",
    }

    return {"service_info": service_info, "model_info": model_info}


prediction_reports = []


@app.get("/reports")
async def get_reports():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"prediction_report_{timestamp}.csv"

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "image_size", "prediction",
                        "timestamp", "execution_time", "model_name"])

        for report in prediction_reports:
            writer.writerow(report)

    return FileResponse(file_name, filename="/reports/prediction_report.csv")


def run_app():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run_app()
