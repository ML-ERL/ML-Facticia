import os
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Configurar la ruta de Tesseract si es necesario
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Cambia según tu sistema si es necesario

def corregir_rotacion(imagen, bbox, angulo):
    """
    Corrige la rotación de una región de interés (ROI) usando el ángulo especificado.

    Args:
        imagen: La imagen original.
        bbox: Las coordenadas de la bounding box en formato YOLO (x_centro, y_centro, ancho, alto).
        angulo: El ángulo de rotación en grados.

    Returns:
        La ROI corregida y alineada.
    """
    x_centro, y_centro, ancho, alto = bbox
    
    # Convertir las coordenadas relativas de YOLO a absolutas
    h, w = imagen.shape[:2]
    x_centro_abs = int(x_centro * w)
    y_centro_abs = int(y_centro * h)
    ancho_abs = int(ancho * w)
    alto_abs = int(alto * h)

    # Definir las coordenadas de la bounding box
    x1 = max(0, x_centro_abs - ancho_abs // 2)
    y1 = max(0, y_centro_abs - alto_abs // 2)
    x2 = min(w, x_centro_abs + ancho_abs // 2)
    y2 = min(h, y_centro_abs + alto_abs // 2)

    # Recortar la región de interés (ROI)
    roi = imagen[y1:y2, x1:x2]

    # Si el ángulo es 0, no es necesario rotar
    if angulo == 0:
        return roi

    # Calcular el nuevo tamaño del lienzo para evitar cortes tras la rotación
    diagonal = int(np.sqrt(roi.shape[0]**2 + roi.shape[1]**2))
    lienzo_expandido = cv2.copyMakeBorder(
        roi,
        top=(diagonal - roi.shape[0]) // 2,
        bottom=(diagonal - roi.shape[0]) // 2,
        left=(diagonal - roi.shape[1]) // 2,
        right=(diagonal - roi.shape[1]) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    # Rotar la ROI
    (h_lienzo, w_lienzo) = lienzo_expandido.shape[:2]
    centro = (w_lienzo // 2, h_lienzo // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    roi_rotada = cv2.warpAffine(lienzo_expandido, matriz_rotacion, (w_lienzo, h_lienzo), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Recortar la ROI rotada al tamaño original de la bounding box
    x_offset = (roi_rotada.shape[1] - ancho_abs) // 2
    y_offset = (roi_rotada.shape[0] - alto_abs) // 2
    roi_final = roi_rotada[y_offset:y_offset + alto_abs, x_offset:x_offset + ancho_abs]
    
    return roi_final

def preprocesar_imagen(imagen):
    """
    Aplica preprocesamiento básico a una imagen para mejorar la extracción de texto.

    Args:
        imagen: La imagen original.

    Returns:
        Imagen preprocesada en escala de grises y binarizada.
    """
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Reducir ruido y aumentar contraste
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binaria

def extraer_texto(imagen, idioma="spa"):
    """
    Extrae texto de una imagen usando Tesseract OCR.

    Args:
        imagen: La imagen preprocesada.
        idioma: El idioma para Tesseract OCR.

    Returns:
        El texto extraído de la imagen.
    """
    # Convertir la imagen a formato compatible con Tesseract
    pil_imagen = Image.fromarray(imagen)
    
    # Configuración avanzada de Tesseract
    config = "--psm 6 --oem 1"
    texto = pytesseract.image_to_string(pil_imagen, lang=idioma, config=config)
    
    return texto

def procesar_carpetas(carpeta_imagenes, carpeta_labels, archivo_salida):
    """
    Procesa todas las imágenes y etiquetas en las carpetas designadas.
    Extrae texto de cada ROI y guarda los resultados en un archivo.

    Args:
        carpeta_imagenes: Ruta de la carpeta con las imágenes.
        carpeta_labels: Ruta de la carpeta con los labels en formato YOLO.
        archivo_salida: Archivo donde se guardarán los resultados.
    """
    # Obtener listas ordenadas de imágenes y labels
    imagenes = sorted(os.listdir(carpeta_imagenes))
    labels = sorted(os.listdir(carpeta_labels))
    
    if len(imagenes) != len(labels):
        raise ValueError("El número de imágenes y labels no coincide.")
    
    resultados = []
    
    for img_file, label_file in zip(imagenes, labels):
        ruta_imagen = os.path.join(carpeta_imagenes, img_file)
        ruta_label = os.path.join(carpeta_labels, label_file)
        
        # Cargar la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"Error al cargar la imagen: {ruta_imagen}")
            continue
        
        # Leer el archivo de etiquetas
        with open(ruta_label, "r") as f:
            etiquetas = f.readlines()
        
        for i, etiqueta in enumerate(etiquetas):
            # Leer los datos de la etiqueta (YOLO: clase, x_centro, y_centro, ancho, alto, angulo)
            datos = list(map(float, etiqueta.strip().split()))
            
            # Si el archivo no incluye ángulo, asignar 0 como predeterminado
            if len(datos) == 5:  # Solo incluye clase, x_centro, y_centro, ancho, alto
                datos.append(0.0)  # Agregar ángulo predeterminado
            
            # Separar los valores
            clase, x_centro, y_centro, ancho, alto, angulo = datos
            
            # Solo procesar clases relevantes (texto, captions, escrito a mano)
            if int(clase) in [1, 3, 4]:
                bbox = (x_centro, y_centro, ancho, alto)
                roi_corregida = corregir_rotacion(imagen, bbox, angulo)
                
                # Preprocesar la ROI
                roi_preprocesada = preprocesar_imagen(roi_corregida)
                
                # Extraer texto
                texto = extraer_texto(roi_preprocesada)
                resultados.append({"archivo": img_file, "texto": texto})
    
    # Guardar resultados en un archivo
    with open(archivo_salida, "w", encoding="utf-8") as archivo:
        for resultado in resultados:
            archivo.write(f"{resultado['archivo']}:\n{resultado['texto']}\n\n")
    
    print(f"Resultados guardados en {archivo_salida}")

# Rutas de carpetas
carpeta_imagenes = "images/"
carpeta_labels = "labels/"
archivo_salida = "resultados_texto.txt"

# Ejecutar el procesamiento
procesar_carpetas(carpeta_imagenes, carpeta_labels, archivo_salida)
