# Propuesta 1

## Análisis de Sentimientos para comprender el tono y emociones en los textos

**Análisis de Sentimientos**

**Objetivo:** Comprender el tono y emociones en los textos de la colección.

**Posibles soluciones:**

- **Modelo Preentrenado de Análisis de Sentimientos:**
Utilizar modelos preentrenados como *BERT* o *RoBERTa* adaptados a análisis de sentimientos. Puedes ajustar estos modelos al corpus específico.
    - **Herramientas:** Hugging Face, SpaCy.
    - **Proceso:** Tokenizar los textos -> Entrenar con etiquetas de sentimientos (positivo, negativo, neutral) si cuentas con un subconjunto etiquetado -> Evaluar con métricas como F1 y precisión.
- **Modelos Supervisados:**
Crear un dataset etiquetado para entrenar un modelo supervisado (Logistic Regression, SVM o CNN simple para texto).
    - **Ventaja:** Más control sobre el modelo.
    - **Desventaja:** Requiere un dataset etiquetado.
- **Aspect-Based Sentiment Analysis (ABSA):**
Si hay textos con opiniones sobre múltiples aspectos (e.g., política, sociedad), este método puede identificar el sentimiento por aspecto.

Papers encontrados:

- [https://paperswithcode.com/task/sentiment-analysis](https://paperswithcode.com/task/sentiment-analysis)
    - **Sentiment Analysis**
    - Resumen:
        
        Este artículo compara diferentes modelos de aprendizaje profundo aplicados al análisis de sentimientos, destacando su capacidad para clasificar polaridades como **positivo**, **negativo**, y **neutral**. Los modelos evaluados incluyen **DNN (Deep Neural Networks)**, **CNN (Convolutional Neural Networks)**, y **RNN (Recurrent Neural Networks)**. Se utilizaron dos técnicas principales de extracción de características: **TF-IDF** y **Word Embedding**, siendo esta última la más efectiva al capturar contextos semánticos complejos.
        
        El estudio empleó datasets como **Sentiment140** (1.6 millones de tweets etiquetados) y reseñas de IMDB, logrando evaluaciones consistentes mediante métricas como **Accuracy, Recall, F1-Score, y AUC**. Entre los resultados destacados:
        
        - **CNN:** Mejor equilibrio entre precisión y tiempo de procesamiento.
        - **RNN:** Mayor fiabilidad y rendimiento al usar Word Embedding, pero con mayor costo computacional.
        - **DNN:** Fácil de implementar, con resultados promedio.
        
        En términos comparativos, las **CNN** demostraron ser ideales para tareas sensibles al tiempo, mientras que las **RNN** sobresalen en aplicaciones donde la comprensión contextual profunda es prioritaria. El artículo concluye que las técnicas basadas en aprendizaje profundo son efectivas para el análisis de sentimientos en áreas como marketing, análisis de productos y monitoreo de redes sociales.
        
- https://paperswithcode.com/paper/convolutional-neural-networks-for-sentence
    - Convolutional Neural Networks for Sentence Classification
    - Resumen:
        
        Este artículo explora el uso de **Redes Neuronales Convolucionales (CNNs)** para tareas de clasificación de oraciones, destacando su capacidad para capturar relaciones contextuales en textos. El objetivo principal es clasificar oraciones en diferentes categorías, como polaridad de sentimientos (positivo/negativo), subjetividad, y tipos de preguntas.
        
        Las CNNs utilizan vectores preentrenados como **word2vec** o **GloVe** para la representación de palabras, junto con técnicas como **max-pooling** para seleccionar las características más relevantes de cada oración. Estas representaciones permiten extraer información contextual dentro de ventanas de palabras específicas.
        
        Se evaluó el rendimiento en diversos datasets:
        
        - **SST-1/SST-2 (Stanford Sentiment Treebank):** Polaridad de sentimientos.
        - **TREC:** Clasificación temática de preguntas.
        - **MR y Subj:** Análisis de opiniones y subjetividad.
        - **MPQA:** Detección de opiniones.
        
        Los resultados muestran que las **CNNs** superan a métodos tradicionales y a algunos modelos avanzados en tareas específicas. Por ejemplo:
        
        - En **SST-2**, las CNNs alcanzaron una precisión del **87.2%** al ajustar los vectores preentrenados.
        - En **Subj**, lograron **93.4%** de precisión.
        
        En términos comparativos, las CNNs destacan por su simplicidad, eficiencia y capacidad para adaptarse a diversos tipos de tareas. Los modelos ajustados (*fine-tuned*) con vectores preentrenados superaron a las configuraciones estáticas, mostrando que el ajuste a un corpus específico mejora significativamente el rendimiento. Este estudio subraya la efectividad de las CNNs en tareas de clasificación de texto frente a métodos tradicionales.
        
- [https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional](https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional)
    - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    - Resumen:
        
        El artículo presenta **BERT (Bidirectional Encoder Representations from Transformers)**, un modelo preentrenado diseñado para tareas de procesamiento del lenguaje natural (NLP) como clasificación de texto, análisis de sentimientos, y más. El principal objetivo es aprender representaciones profundas bidireccionales del contexto, capturando relaciones antes y después de cada palabra en una oración.
        
        El modelo se entrena en tareas no supervisadas como:
        
        - **Máscara de palabras (Masked Language Model - MLM):** Predicción de palabras ocultas en una oración.
        - **Next Sentence Prediction (NSP):** Determinación de la relación entre dos oraciones consecutivas.
        
        BERT se evaluó en múltiples datasets y tareas, incluidas:
        
        - **SST-2:** Clasificación binaria de sentimientos.
        - **MNLI, QNLI:** Clasificación de relaciones entre textos.
        - **SQuAD:** Respuesta a preguntas basada en texto.
        
        **Resultados destacados:**
        
        - En **SST-2**, BERT alcanzó una precisión del **94.9%**, superando modelos como OpenAI GPT.
        - En tareas de comprensión lectora como **SQuAD**, logró **92.7% de F1** y **84.1% de exactitud**, estableciendo un nuevo estado del arte.
        
        Comparado con otros modelos, BERT sobresale por:
        
        - Su enfoque bidireccional, que mejora la captura de contexto.
        - Su flexibilidad para adaptarse a diversas tareas con un ajuste (*fine-tuning*) eficiente.
        
        Este trabajo destaca cómo los modelos preentrenados como BERT revolucionaron el NLP, logrando resultados superiores en tareas clave con una arquitectura que aprovecha la profundidad y la bidireccionalidad de los transformadores.
        
- [https://paperswithcode.com/paper/roberta-a-robustly-optimized-bert-pretraining](https://paperswithcode.com/paper/roberta-a-robustly-optimized-bert-pretraining)
    - RoBERTa: A Robustly Optimized BERT Pretraining Approach
    - Resumen:
        
        El artículo presenta **RoBERTa**, una versión optimizada de BERT que mejora significativamente su rendimiento mediante ajustes en el proceso de preentrenamiento. El objetivo principal es demostrar que el preentrenamiento más robusto y con mayores recursos puede mejorar los resultados sin modificar la arquitectura base de BERT.
        
        Las optimizaciones incluyen:
        
        - **Eliminación de Next Sentence Prediction (NSP):** Se omite esta tarea, permitiendo concentrar más recursos en la predicción de palabras enmascaradas.
        - **Aumento del tamaño del corpus:** RoBERTa se entrena en un conjunto de datos de mayor escala (160 GB frente a 16 GB en BERT).
        - **Mayor tiempo de entrenamiento y lotes más grandes.**
        
        **Datasets evaluados:**
        
        - RoBERTa se probó en tareas de clasificación como **SST-2** (análisis de sentimientos), **MNLI** (inferencia textual), y otros conjuntos del benchmark GLUE, así como en tareas de comprensión lectora como **SQuAD**.
        
        **Resultados destacados:**
        
        - En **SST-2**, alcanzó una precisión del **96.4%**, superando a BERT y otros modelos de su categoría.
        - Mejoró en todas las tareas del GLUE Benchmark, estableciendo nuevos estados del arte en varias de ellas.
        - En **SQuAD v1.1**, obtuvo un **F1 del 94.6%**, consolidando su rendimiento en tareas de comprensión lectora.
        
        **Comparación con BERT:**
        
        - RoBERTa utiliza el mismo diseño arquitectónico que BERT, pero sus optimizaciones en el preentrenamiento lo hacen más preciso y eficiente.
        - Es especialmente efectivo en tareas que requieren grandes cantidades de datos y mayor capacidad de representación contextual.
        
        El trabajo concluye que los ajustes en el preentrenamiento son clave para mejorar el rendimiento de modelos de lenguaje, y RoBERTa demuestra que las capacidades de BERT pueden potenciarse con estrategias de entrenamiento más robustas.
        
- https://paperswithcode.com/paper/bag-of-tricks-for-efficient-text
    - Bag of Tricks for Efficient Text Classification
    - Resumen:
        
        El artículo presenta un enfoque práctico y eficiente para la clasificación de texto utilizando **fastText**, una herramienta basada en métodos simples pero poderosos. El objetivo principal es proporcionar técnicas que combinen simplicidad y velocidad con resultados competitivos frente a modelos más complejos.
        
        **Modelos utilizados:**
        
        - fastText, que se basa en una representación de palabras mediante **Bag of Words (BoW)** y **n-gramas** para capturar características locales.
        
        **Técnicas de extracción de características:**
        
        - **TF-IDF:** Usada para representar la relevancia de las palabras.
        - **Word n-grams:** Agregados al modelo para mejorar la representación del contexto local.
        
        **Datasets evaluados:**
        
        - **Yelp:** Clasificación de reseñas positivas y negativas.
        - **Amazon:** Análisis de productos.
        - **AG News:** Clasificación de artículos noticiosos en temas generales como política, negocios y tecnología.
        - **Yahoo Answers:** Categorización de preguntas y respuestas en diversas temáticas.
        
        **Resultados destacados:**
        
        - **Yelp:** Alcanzó una precisión del **95.7%** con n-gramas.
        - **Amazon:** Precisión del **94.6%** en clasificación de sentimientos.
        - **AG News:** Precisión cercana al **92%**, comparable a redes neuronales profundas.
        
        **Comparación con otros modelos:**
        
        - **Velocidad:** fastText es significativamente más rápido que modelos avanzados como CNNs o RNNs, siendo **10-15,000 veces más eficiente** en entrenamiento y predicción.
        - **Rendimiento:** Aunque más simple, logra precisión comparable a métodos avanzados gracias a n-gramas y su arquitectura optimizada.
        
        El artículo concluye que fastText es una solución ideal para tareas de clasificación de texto en escenarios donde la velocidad y la simplicidad son prioritarias, sin comprometer significativamente la precisión. Su enfoque lo hace especialmente útil para aplicaciones prácticas con grandes volúmenes de datos