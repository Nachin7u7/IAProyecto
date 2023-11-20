# IAProyecto - VISA INSPECTOR

Verificador de seguridad hogareña donde se reciben imagenes tomadas de una camara de seguridad del hogar y te pregunta si eres tu el que esta rondando dentro de la casa, caso contrario te dira que no hay nada de que preocuparse.

#### Que se uso

Se utilizo Yolov3 (Reconocimiento de personas), FastAPI, Uvicorn

### Contenido (endpoints)

**GET/status:** muestra datos del servidor, si es que esta corriendo y que archivos se esta usando para el motor de reconocimiento.  
**POST/predict:** se sube la imagen y da el resultado sobre si se encuentra alguien dentro de la casa o no y registra el tiempo, informacion de la imagen, etc.  
**GET/reports:** nos da la positibiladad de descargar un archivo CSV con todas las capturas que se hizo, con la informacion como la hora, tamaño, etc.

### Como ejecutar el programa

correr la app en localhost/docs y subir la imagen

archivos necesarios de la propia libreria de YOLO version 3 o 4, donde se agregaron yolov3.cfg, yolov3.weights (<--- esta siendo informacion preentrenada con distintos tipos de objetos y un accurancy alto), coco.names

para descargar yolov3.weights se puede encontrar en el repositorio oficial de yolo (pesa un poco mas de 200 Mb)

Descarga de yolov3.weights ---> https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights

y colocar el archivo en la carpeta raiz, acto seguido ejecutar el programa

#### Respuestas
##### `POST /predict/`
- **200 OK**
  - Descripción: La predicción fue exitosa.
  - Ejemplo:
    ```json
    {
      "result": "All clear, nothing to worry by now.",
      "execution_time": 0.123
    }
    ```

- **200 OK**
  - Descripción: Se detectó la presencia de un objeto (clase 0, que representa a personas en el conjunto de datos COCO).
  - Ejemplo:
    ```json
    {
      "result": "There is an entity at home, is that you?",
      "execution_time": 0.456
    }
    ```

#### Ejemplo de Uso (cURL)
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
```

##### `GET /status`

200 OK
Descripción: Información sobre el estado del servicio y del modelo.  
Ejemplo:

```json
{
  "service_info": {
    "status": "running",
    "message": "Service is up and running."
  },
  "model_info": {
    "model_name": "YOLOv3",
    "weights_file": "yolov3.weights",
    "config_file": "yolov3.cfg",
    "classes_file": "coco.names"
  }
}
```
#### Ejemplo de Uso (cURL)
```bash
curl -X GET "http://127.0.0.1:8000/status" -H "accept: application/json"
```

##### `GET /reports`

200 OK  
Descripción: Archivo CSV con información sobre las predicciones.  
Ejemplo: _Descarga del archivo prediction_report.csv._

#### Ejemplo de Uso (cURL)
```bash
curl -X GET "http://127.0.0.1:8000/reports" -H "accept: application/csv"
```

## Estudiantes:
#### Ricardo I. Valencia  
#### Alejandra Garcia
