# Colección Facticia - Digitalización de Documentos

Este proyecto tiene como objetivo digitalizar la colección facticia de Emilio Roig de Leuchsenring, extrayendo imágenes y textos de documentos antiguos mediante técnicas de deep learning y OCR. Se utilizan modelos especializados para cada tarea:

- **Detección de Objetos:** YOLO v8 para identificar y recortar imágenes de interés.
- **OCR:** Tesseract para extraer textos de las imágenes.
- **Similitud Imagen-Texto:** CLIP para emparejar imágenes con sus descripciones.

## Estructura del Proyecto

El proyecto cuenta con 4 carpetas principales:

- **dataset**: Almacena los datos de entrenamiento en caso de existir, además de la carpeta por defecto con las imágenes a procesar y la ubicación para almacenar los crops
- **experimentation**: Contiene los procesamientos realizados para determinar las mejores alternativas de preprocesamiento para Tesseract
- **src**: Contiene todos los archivos de la lógica del programa, entre ellos la definición de la clase full model que integra los diversos modelos y cada una de las implementaciones específicas
- **training models**: Contiene la información de los artículos consultados para el desarrollo de este proyecto

En el directorio raíz podemos encontrar:

- **app.py**: punto de entrada a la aplicación de streamlit
- **main.py**: punto de entrada a la aplicación de consola

## Uso

**Instalación:**  
   Instalar las dependencias listadas en `requirements.txt`.  
   ```bash
   pip install -r requirements.txt
   ```

## Streamlit

[Drive con imágenes de ejemplo](https://drive.google.com/drive/folders/1mGBljfrRwHWtKwOliC1mbVcq7K0LCMV9?usp=sharing)

[Sreamlit](https://ml-facticia.streamlit.app/)

Pasos a seguir:
1. Presionar el botón **Upload File** y seleccionar el archivo a procesar
2. Se muestra una previzualización de la imagen
3. Se ejecuta el modelo de detección y se muestran los recortes extraidos
4. En el caso de las imágenes se muestra información de los captions cercanos y los textos más similares