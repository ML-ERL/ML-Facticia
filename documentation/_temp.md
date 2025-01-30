## Por qué usar CLIP?

Clip es un modelo que aprende a asociar imágenes y sus correspondientes descripciones textuales entrenándose en un gran conjunto de datos de pares imagen-texto. En lugar de predecir etiquetas específicas CLIP aprende un espacio donde las imagenes se encuentran cercanas a sus descripciones de texto y alejadas de descripciones de texto no muy relacionadas. Esto permite al modelo generalizar para datos que no ha visto nunca sin necesidad de entrenamiento específico para la tarea (**Zero-Shot**). El modelo utiliza redes neuronales (**Transformers**) para extraer las *características* visuales y textuales y asignar a cada entrada una ubicación en el espacio de **embedding** con la misma dimensionalidad. Por estos motivos decidimos utilizar este modelo para determinar la proximidad entre las imágenes y los textos extraidos, aprovechando su capacidad para establecer un puente entre vision y lenguaje, y, por otra parte, reducir considerablemente el tiempo de desarrollo al no requerir un entrenamiento previo para la obtención de resultados relevantes.

## Resultados

En el procesamiento de los datos con que se utiliza el modelo se utilizaron dos métricas para la detección de similitud: distancia de Jaccard y **Character Error Rate** (CER). En el primer caso podemos obtener una buena aproximación de que tan fiel es el texto extraído con respecto al texto original, teniendo en cuenta las palabras que aparecen en cada uno con un costo computacional bastante acotado. Después de algunas iteraciones se decidió utilizar CER debido a que para la construcción de los **embeddings** Clip utiliza **Byte Pair Encoding** (BPE) el cual empieza con un vocabulario base de caracteres, mezcla iterativamente los pares más comunes de caracteres o subpalabras para formar nuevas subpalabras y de este modo termina con un vocabulario con un balance entre ser basado en caracteres y en palabras.


## Por qué usar Tesseract?

Tesseract es un modelo actual que implementa técnicas que son parte del Estado del Arte en el campo de la conversión de texto a imagen. Se encuentra preentrenado para datos en español por lo que puede ser utilizado en el contexto de nuestro proyecto sin necesidad de agregar nuevos datos de entrenamiento. Entre sus potencialidades se encuentra que su uso es libre de costo y al ser un proyecto **open source** obtiene grandes beneficios al recibir contribuciones y mejoras por parte de la comunidad. Tesseract recibe actualizaciónes periodicas e incorpora técnicas avanzadas para el OCR como los enfoques basados en **deep-learning**.

## Resultados

Se realizaron varias pruebas para determinar que preprocesamientos daban mejores resultados con Tesseract. A continuación los resultados obtenidos:

* insertar tabla

Por este motivo se decidió utilizar la combinación de las técnicas ... 

## Mejorar los resultados usando un llm

Por la naturaleza de los datos, los modelos de reconocimiento de texto en imagenes suelen tener algunos problemas, es por ello que proponemos como parte de nuestra solución mejorar los resultados obtenidos utilizando un modelo de lenguaje para esta tarea.

Notas:

- Decidimos priorizar la extracción con calidad de los crops