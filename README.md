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


## Estudiantes:
### Ricardo I. Valencia
### Alejandra Garcia
