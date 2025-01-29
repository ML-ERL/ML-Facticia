# Propuesta 2

## Clasificación temática para identificar temas recurrentes y categorizar documentos agrupados en grandes temas generales (política, sociedad, economía, etc.).

- https://arxiv.org/abs/2306.02864?utm_source=chatgpt.com
    - **Nombre**: Leveraging Large Language Models for Topic Classification in the Domain of Public Affairs
    - **Fecha**: 2023
    - **Resumen**:
        
        Este artículo explora el uso de **Modelos de Lenguaje a Gran Escala (LLMs)** para la clasificación temática en el dominio de los asuntos públicos, centrándose en categorizar documentos en temas generales como salud, cambio climático, y sostenibilidad. El objetivo principal es evaluar la efectividad de modelos preentrenados en la identificación de temas relevantes en textos gubernamentales y legislativos.
        
        ## Secciones principales:
        
        ### **1 Data Collection and Analysis**
        
        - **Corpus:** El conjunto de datos utilizado comprende más de **33,000 documentos** relacionados con políticas legislativas, regulaciones gubernamentales y reportes públicos.
        - **Etiquetas:** Cada documento fue categorizado manualmente en **30 temas principales**.
        - **Preprocesamiento:** Se llevaron a cabo técnicas estándar como eliminación de stopwords, tokenización, y lematización para limpiar y preparar el texto para los modelos.
        - **Distribución temática:** Los temas presentan una distribución desbalanceada, lo que plantea desafíos para la clasificación, especialmente en clases con menos datos.
        
        ### **2 Methodology and Models**
        
        - **Modelos principales:**
            - **RoBERTa-base y RoBERTa-large:** Se utilizan como LLMs preentrenados para capturar representaciones contextuales profundas.
            - **GPT-2:** Utilizado como modelo generativo para la comprensión del texto.
            - **Herramientas adicionales:**
                - **SVM (Support Vector Machines):** Clasificador adicional para las representaciones extraídas por RoBERTa.
                - Redes neuronales y Random Forest como comparadores.
        - **Enfoque híbrido:**
            - RoBERTa proporciona embeddings textuales, que luego se clasifican utilizando SVM o redes neuronales.
            - Esta combinación permite aprovechar la potencia contextual de los LLMs y la precisión de los clasificadores lineales.
        - **Técnicas complementarias:**
            - Se utiliza validación cruzada con 5 pliegues para garantizar la consistencia de los resultados.
            - Métricas de evaluación como precisión, recall y F1-score para medir el rendimiento.
            
            ### **Técnicas de extracción de características**
            
            - Los modelos emplearon representaciones contextuales avanzadas basadas en **word embeddings**, optimizando la comprensión semántica a través de sus arquitecturas preentrenadas.
        
        ### **3 Experiments**
        
        - **Configuración experimental:**
            - Los modelos fueron entrenados y evaluados en divisiones de entrenamiento/prueba del conjunto de datos.
            - Se realizaron comparaciones con métodos tradicionales como Naive Bayes y Logistic Regression.
        - **Resultados destacados:**
            - **RoBERTa + SVM:** Logró una precisión promedio superior al **85%**, mostrando el mejor rendimiento en la mayoría de los temas, incluidas categorías con pocos datos.
            - **GPT-2:** Desempeño competitivo, aunque con mayor tiempo de procesamiento.
            - **SVM tradicional:** Aunque rápido, presentó dificultades para manejar la complejidad semántica de los textos.
            - **Análisis por tema:** Los temas con datos más balanceados mostraron métricas superiores, mientras que las clases desbalanceadas presentaron desafíos.
            - **Comparación:**
                - Los modelos preentrenados como RoBERTa destacan por su capacidad de generalización y manejo de múltiples temas simultáneamente.
                - SVM mejoró la clasificación al combinarse con RoBERTa, superando métodos tradicionales como Random Forest.
        - **Visualizaciones:** Gráficos de matriz de confusión y curvas ROC para demostrar el rendimiento de los modelos.
        
        **4 Conclusions**
        
        - **Rendimiento de los LLMs:** RoBERTa demostró ser una herramienta poderosa para capturar la complejidad semántica en documentos públicos, superando significativamente los enfoques tradicionales.
        - **Aplicaciones prácticas:** El trabajo destaca la escalabilidad de los LLMs para clasificaciones temáticas en dominios especializados y propone mejoras futuras mediante ajustes específicos al dominio (domain-specific fine-tuning).
        - **Líneas futuras:** Optimizar el modelo para manejar mejor clases desbalanceadas y explorar técnicas de fine-tuning específicas por dominio.
        
        ### 
        
- https://arxiv.org/abs/2406.14983?utm_source=chatgpt.com
    - **Nombre**: Hierarchical Thematic Classification of Major Conference Proceedings
    - **Fecha**: 2024
    - **Resumen**:
        
        Este artículo presenta un sistema de apoyo a la decisión para la clasificación jerárquica de texto en colecciones con estructuras temáticas predefinidas. Su objetivo principal es clasificar documentos jerárquicamente, asignándolos a temas relevantes dentro de una estructura en forma de árbol, facilitando la búsqueda y organización eficiente de colecciones de textos como resúmenes científicos y sitios web. Se centra en identificar temas recurrentes y categorizar documentos en grandes áreas generales dentro de jerarquías predefinidas. Las aplicaciones incluyen áreas como política, sociedad y economía en colecciones estructuradas.
        
        ## Secciones principales:
        
        ### **1 Weighted Hierarchical Similarity Function**
        
        - **Función de similitud ponderada (hSim):**
            - Diseñada para medir la relevancia de un documento en relación con un nodo en una jerarquía temática considerando la importancia de las palabras basada en su entropía.
            - Cada nodo jerárquico incluye pesos específicos basados en la frecuencia y entropía de las palabras en el corpus.
            - Se integran métricas como la similitud del coseno, ajustadas para considerar la jerarquía y las relaciones padre-hijo.
        - **Ventaja principal:** hSim no solo evalúa similitudes en el nivel del nodo, sino también en toda la rama jerárquica, mejorando la asignación en estructuras profundas.
        
        ### **2 Model and Parameters Estimation**
        
        - **Modelo jerárquico:**
            - Cada documento es evaluado en función de su probabilidad de pertenecer a un nodo específico.
            - Se utilizan representaciones de texto basadas en TF-IDF para calcular similitudes iniciales.
        - **Estimación de parámetros:**
            - Los parámetros de relevancia de los nodos se ajustan en función de la distribución de las palabras en el corpus.
            - Algoritmos iterativos optimizan los pesos de los nodos para maximizar la precisión de la clasificación.
        - **Otros métodos comparados:**
            - Naive Bayes jerárquico (hNB).
            - PLSA con regularización ARTM.
            - SVM multicategoría jerárquico.
            
        
        ### **Técnicas de extracción de características**
        
        - **Peso de las palabras:** Se calcula mediante un modelo de entropía que evalúa su importancia para separar clústeres en la jerarquía.
        - **Medida de similitud:** Basada en coseno ponderado, adaptado para estructuras jerárquicas.
            
            
        
        ### **3 Bayesian Approach in Parameters Estimation**
        
        - **Enfoque bayesiano:**
            - Introduce un modelo probabilístico para ajustar los parámetros de la función de similitud ponderada.
            - Priorización basada en la distribución previa de las palabras en los nodos jerárquicos.
        - **Ventajas:**
            - Mejora la robustez del modelo frente a clases desbalanceadas.
            - Permite integrar nueva información (por ejemplo, documentos adicionales) sin reentrenar el modelo completo.
                
                
        
        ### **4 Computational Experiment**
        
        - **Dataset:**
            - **Abstracts del EURO Conference:**
                - 15,527 documentos.
                - organizados en una jerarquía de **26 temas principales** y **264 subtemas**
            - **Sitios web industriales:**
                - 1,036 documentos con jerarquías específicas de clústeres.
        - **Resultados:**
            - La función hSim alcanzó un **AUCH (Área bajo el histograma acumulativo)** de **0.93**, superando a métodos como hNB (Naive Bayes jerárquico) (**0.92**) y SuhiPLSA (una variante de PLSA)(**0.84**).
            - En los sitios web industriales, hSim logró un AUCH de **0.89**, superior a SVM (**0.83**) y hNB (**0.83**).
            - En términos de precisión, la hSim mostró una mejora significativa en ramas más profundas de la jerarquía, donde los métodos tradicionales suelen fallar.
        - **Comparaciones:**
            - hSim es más computacionalmente eficiente que SuhiPLSA, mostró mejores resultados en todos los experimentos, gracias a su capacidad para incorporar pesos de palabras y evaluar ramas completas en lugar de depender exclusivamente de enfoques jerárquicos tradicionales (*top-down*).
            - Métodos como hNB y SVM fueron competitivos, pero con limitaciones al manejar estructuras más profundas.
            
            ### **5 Conclusion**
            
            - **Contribuciones principales:**
                - Propuesta de la función hSim como una solución robusta para la clasificación jerárquica.
                - Incorporación de un enfoque bayesiano que mejora la estimación de parámetros y maneja eficientemente las clases desbalanceadas.
            - **Impacto:** Este modelo establece un marco replicable para sistemas jerárquicos de clasificación en dominios complejos como conferencias académicas, catálogos temáticos y bibliotecas digitales.
            - **Futuras direcciones:** Extender el modelo a estructuras más complejas como grafos, e integrar técnicas de deep learning para mejorar las representaciones de texto.
            
- https://www.sciencedirect.com/science/article/abs/pii/S0306457318307805
    - **Nombre**: An Evaluation of Document Clustering and Topic Modelling in Two Online Social Networks: Twitter and Reddit
    - **Fecha**: 2020
    - **Resumen**:
        
        Este artículo evalúa métodos de agrupamiento de documentos y modelado de temas en redes sociales, específicamente en **Twitter** y **Reddit**, para identificar temas recurrentes y categorizar textos. El objetivo principal es comparar enfoques de **clustering** y **modelado de temas** para descubrir automáticamente las principales áreas temáticas en los datos generados por los usuarios.
        
        ### **2. Literature Review**
        
        - **Agrupamiento de documentos:**
            - Se revisan métodos tradicionales como **k-means** y **hierarchical clustering**, destacando sus limitaciones en contextos de datos desestructurados como redes sociales.
            - Se analiza el impacto del uso de técnicas como **Latent Dirichlet Allocation (LDA)** y **Non-negative Matrix Factorization (NMF)** para el modelado de temas.
        - **Desafíos identificados:**
            - Heterogeneidad y ruido en los datos sociales.
            - Dificultades para manejar el alto volumen y la variabilidad temática de los datos.
        
        ### **3 Methods**
        
        1. **Datos utilizados:**
            - **Twitter:** Publicaciones centradas en eventos de alto impacto, como campañas políticas.
            - **Reddit:** Discusiones de subreddits específicos, categorizadas en diferentes temas.Ambos datasets incluyen textos cortos y largos, facilitando la evaluación en diferentes contextos.
        2. **Preprocesamiento:**
            - Eliminación de stopwords, tokenización, y lematización.
            - Filtrado de publicaciones irrelevantes.
        3. **Técnicas aplicadas:**
            - **Modelado de temas:** **Latent Dirichlet Allocation (LDA) y NMF:** Utilizado para extraer temas recurrentes y asignar probabilidades temáticas a cada documento.
            - **Clustering:**
                - **k-means:** Agrupa documentos en clústeres basados en sus características textuales.
                - **Clustering Jerárquico:** Una técnica jerárquica para construir relaciones entre documentos y temas.
            - **Otros métodos:** Se evalúan métricas de similitud como coseno para mejorar los resultados del agrupamiento.
            - **Métricas de evaluación:** Silhouette score, coherencia de temas, y análisis visual para medir la calidad de los clústeres y los temas generados.
        4. **Técnicas de extracción de características**
            - **TF-IDF:** Representación básica de los textos para capturar la relevancia de las palabras en el corpus.
            - **Word embeddings:** Empleados en ciertos experimentos para capturar relaciones semánticas más profundas entre términos.
        
        ### **Results**
        
        - **LDA vs. NMF:**
            - **LDA:** Identificó temas con mayor coherencia semántica, especialmente en Reddit, donde las discusiones son más estructuradas.
            - **NMF:** Más eficiente en términos computacionales, pero con menor coherencia temática.
        - **Clustering:**
            - k-means funcionó bien para agrupar temas generales en ambos datasets, pero tuvo dificultades con categorías específicas en Twitter debido al ruido,menos preciso en la representación de relaciones semánticas.
            - El clustering jerárquico mostró un mejor rendimiento al capturar relaciones entre temas en Reddit, adecuado para descubrir relaciones jerárquicas entre temas, aunque más costoso computacionalmente.
        - **Resultados por red social:**
            - **Twitter:** Mayor ruido en los datos, lo que dificultó la extracción de temas claros.
            - **Reddit:** Produjo clústeres más precisos debido a su estructura temática más definida.
        - **Comparación:**
            - LDA fue más eficaz en tareas de modelado de temas, mientras que k-means sobresalió en eficiencia y simplicidad.
            - Reddit mostró mejor desempeño en consistencia temática debido a su estructura más organizada en comparación con los datos no estructurados de Twitter.
        
        ### **Conclusión**
        
        El estudio concluye que combinar modelado de temas (LDA) con técnicas de clustering mejora la calidad de la clasificación temática, especialmente en plataformas con estructuras más definidas como Reddit. Este enfoque permite identificar temas recurrentes y categorizar documentos en grandes áreas como política, sociedad y economía.
        
- [https://arxiv.org/html/2410.03721v1](https://arxiv.org/html/2410.03721v1)
    - **Nombre**: Thematic Analysis with Open-Source Generative AI and Machine Learning
    - **Fecha**: 2024
    - **Resumen**:
        
        Este artículo presenta un nuevo enfoque para el análisis temático utilizando inteligencia artificial generativa y aprendizaje automático. El objetivo principal es facilitar la identificación y clasificación de temas en grandes volúmenes de texto mediante el desarrollo del flujo de trabajo **GATOS (Generative AI-enabled Theme Organization and Structuring)**, orientado al desarrollo inductivo de códigos cualitativos.
        
        ## Secciones principales:
        
        ### **Dataset**
        
        - Textos obtenidos de entrevistas, encuestas y documentos organizacionales, utilizados para construir y validar el flujo de trabajo. Los datos abarcan temas generales y específicos en áreas como trabajo en equipo, liderazgo, y dinámicas organizacionales.
        
        ### **Method**
        
        1. **Plataformas y herramientas utilizadas:**
            - **Modelos generativos:** Basados en GPT para sugerir temas y relaciones entre textos.
            - **Herramientas de código abierto:** Como spaCy para análisis de texto y scikit-learn para clustering.
        2. **Pasos del enfoque:**
            - **Preprocesamiento de datos:** Limpieza de textos, tokenización y eliminación de ruido.
            - **Análisis inicial:** Extracción de términos clave utilizando TF-IDF y embeddings semánticos para representar textos y detectar patrones contextuales y cálculo de similitudes entre documentos para agruparlos en temas relacionados.
            - **Modelado de temas:** Aplicación de técnicas como Latent Dirichlet Allocation (LDA) para identificar patrones.
            - **Clustering:** Agrupación de documentos temáticamente similares mediante k-means.
        3. **Validación:** Revisión manual por expertos cualitativos para evaluar la coherencia de los temas generados
        4. **Técnicas de aprendizaje automático:**
            - **k-means:** Para agrupar textos en clústeres temáticos.
            - Modelos supervisados para validar las etiquetas generadas.
        
        ### **Results**
        
        - **Identificación de temas:**
            - Los modelos generativos identificaron temas recurrentes y propusieron códigos iniciales con una coincidencia promedio del **85%** con códigos manualmente generados.
            - Los clústeres temáticos mostraron una coherencia interna alta, especialmente en textos homogéneos.
        - **Comparación con métodos tradicionales:**
            - El enfoque propuesto redujo el tiempo necesario para desarrollar un código temático en un **50%**.
            - Se destacaron mejoras en la detección de temas latentes no identificados manualmente.
        - **Visualizaciones:** Mapas de temas y redes conceptuales facilitaron la comprensión de relaciones entre códigos.
        - **Comparación entre métodos:**
            - GATOS superó a los métodos manuales y a los modelos tradicionales en tiempo y consistencia, reduciendo significativamente el tiempo necesario para el análisis temático.
        
        ### **Conclusión**
        
        El flujo de trabajo GATOS combina lo mejor de los modelos generativos y las técnicas de aprendizaje automático para ofrecer una solución eficaz a la clasificación temática. Este enfoque es ideal para categorizar documentos en áreas como política, sociedad y economía, proporcionando una metodología reproducible para el análisis cualitativo a gran escala.
        

- https://redc.revistas.csic.es/index.php/redc/article/view/1517
    - **Nombre**: Clasificación temática automática de documentos basada en vocabularios y frecuencias de uso. El caso de artículos de divulgación científica
    - **Fecha**: 2023
    - **Resumen**:
        
        Este artículo propone un sistema automático para la clasificación temática de documentos, enfocándose en artículos de divulgación científica. El objetivo principal es asignar temas a documentos utilizando un enfoque basado en **vocabularios temáticos predefinidos** y **frecuencias léxicas**, logrando una clasificación eficiente y precisa en categorías amplias como ciencia, tecnología, y sociedad.
        
        ### **Enfoque Propuesto**
        
        - **Objetivo principal:** Automatizar la clasificación temática de artículos de divulgación científica mediante el uso de vocabularios temáticos diseñados específicamente para cada categoría.
        - **Etapas del enfoque:**
            1. **Construcción de vocabularios temáticos:**
                - Se crean listas de términos clave asociados a categorías generales como ciencia, tecnología, y sociedad, basándose en análisis previos del corpus.
            2. **Frecuencias léxicas:**
                - Se calcula la frecuencia de aparición de cada término en el vocabulario dentro de los documentos.
                - Las frecuencias son ponderadas por métricas como **TF-IDF** para priorizar palabras más relevantes.
            3. **Asignación de categorías:**
                - Cada documento es asignado a una o más categorías según la suma ponderada de las frecuencias de los términos relevantes que contiene.
                - Las palabras clave de los vocabularios se ponderan según su relevancia para cada tema, optimizando la asignación temática.
        - **Ventajas del enfoque:**
            - Simplicidad y eficiencia computacional, adecuado para corpus medianos y pequeños.
            - No requiere entrenamiento extensivo, lo que lo hace ideal para contextos con recursos limitados.
        
        ### **Modelos utilizados y herramientas adicionales**
        
        - **Modelo de clasificación léxica:** Utiliza vocabularios temáticos predefinidos que asocian palabras clave a categorías específicas.
        - **Algoritmos supervisados:** Clasificadores como Naive Bayes y Support Vector Machines (SVM) se usan como puntos de comparación para evaluar el rendimiento del sistema.
        
        ### **Validación**
        
        - **Corpus utilizado:**
            - Un conjunto de más de **1,000 artículos de divulgación científica en español**.
            - Cada documento fue previamente etiquetado manualmente para evaluar la precisión del enfoque automático.
            - Los vocabularios temáticos se desarrollaron a partir de las palabras más frecuentes en cada categoría.
        - **Resultados:**
            - **Precisión promedio:** Entre **80-85%** en la clasificación automática, dependiendo de la categoría temática.
            - **Categorías específicas:** Las categorías con vocabularios más especializados (e.g., ciencias naturales) lograron mejores resultados que aquellas con mayor superposición de términos (e.g., sociedad).
        - **Comparación con otros métodos:**
            - El enfoque basado en vocabularios obtuvo resultados similares a algoritmos supervisados como Naive Bayes, pero con menor complejidad computacional.
            - Los algoritmos supervisados mostraron ventajas en categorías con datos abundantes, pero el enfoque propuesto superó en clases con datos limitados.
            - SVM mostró una ligera ventaja en precisión, especialmente en categorías con alta superposición de términos, pero a un mayor costo computacional, son más efectivos para datasets grandes o cuando los temas tienen menos palabras clave específicas.
        
        ### **Conclusión**
        
        El enfoque basado en vocabularios y frecuencias de uso léxico es una herramienta efectiva para la clasificación temática, especialmente en contextos donde los recursos computacionales son limitados. Este método facilita la identificación de temas recurrentes y la categorización de documentos en áreas generales como ciencia y sociedad, destacándose por su equilibrio entre precisión y simplicidad.
        
- https://www.proquest.com/openview/92f8689605326f6716da2c05a25c5fe0/1?pq-origsite=gscholar&cbl=4433095
    - **Nombre**: A Thematic Analysis of English and American Literature Works Based on Text Mining and Sentiment Analysis
    - **Fecha**: 2024
    - **Resumen**:
        
        El artículo propone un marco innovador para analizar temas y sentimientos en obras literarias angloamericanas, empleando minería de texto y análisis de sentimientos. Este análisis combina métodos computacionales y técnicas multimodales para obtener patrones temáticos y emocionales. A continuación, se analizan las secciones clave.
        
        ---
        
        ### **Literature Review**
        
        - **Objetivo de la revisión:**
            - Resumir los avances en análisis temático y sentimental de textos literarios.
            - Destacar cómo las técnicas de minería de texto han evolucionado para abordar tareas complejas en literatura.
        - **Técnicas mencionadas:**
            - **TF-IDF y LDA (Latent Dirichlet Allocation):** Usados comúnmente para modelado de temas.
            - **Análisis de sentimientos supervisado y no supervisado:** Comparación de enfoques léxicos y basados en aprendizaje automático.
        - **Limitaciones de trabajos previos:**
            - Falta de integración entre análisis temático y sentimental.
            - Escasa exploración de características multimodales como imágenes o metadatos en literatura.
        - **Contribución del estudio:**
            - Propone una solución que combina temas y emociones, incluyendo información multimodal.
        
        ---
        
        ### **Multimodal Stop Word Extraction in American Literature**
        
        - **Método propuesto:**
            - Desarrollo de un sistema para extraer stopwords específicas a partir de textos literarios angloamericanos.
            - **Stopwords multimodales:** Se identifican términos redundantes en texto, imágenes y audio.
            - **Técnicas utilizadas:**
                - **TF-IDF adaptado:** Ajustado para ignorar palabras con alta frecuencia pero baja relevancia en contexto literario.
                - **Extracción jerárquica:** Analiza capítulos completos para identificar términos redundantes específicos a cada obra.
        - **Resultados:**
            - Mejora en la representación semántica al eliminar palabras irrelevantes específicas del dominio.
            - Incremento del **12% en la coherencia temática** al utilizar datos procesados sin stopwords multimodales.
        
        ---
        
        ### **Bi-Gram Multimodal Sentimental Analysis (Bi-gramMSA)**
        
        - **Descripción del modelo:**
            - Analiza pares consecutivos de palabras (**bi-gramas**) para identificar patrones emocionales y contextuales.
            - Integra datos multimodales, incluyendo texto, imágenes y audio.
        - **Modelo utilizado:**
            - **Redes neuronales convolucionales (CNN):** Para capturar relaciones semánticas y emocionales entre bi-gramas.
            - **Algoritmo de análisis de sentimientos supervisado:** Entrenado con un corpus literario anotado manualmente.
        - **Características extraídas:**
            - **Frecuencias de bi-gramas:** Representan combinaciones comunes de palabras y sus asociaciones emocionales.
            - **Embeddings textuales:** Generados a partir de Word2Vec para bi-gramas.
            - **Características visuales:** Extraídas de ilustraciones en libros mediante CNN.
        - **Resultados:**
            - **Precisión en análisis de sentimientos:** 88.7% en predicción de polaridad.
            - **Mejora en modelado de temas:** Identificación más precisa de relaciones entre emociones y temas literarios.
        
        ---
        
        ### **6. Conclusion**
        
        - **Aportaciones principales:**
            - Propuesta de un marco que combina análisis temático y sentimental en literatura.
            - Innovación en la extracción de stopwords multimodales y uso de bi-gramas para análisis emocional.
        - **Resultados generales:**
            - Mejora significativa en precisión y coherencia al integrar datos multimodales.
            - Incremento del **20% en la identificación de patrones temáticos** en comparación con enfoques unidimensionales.
        - **Impacto del trabajo:**
            - Aplicaciones potenciales en análisis literario, educación y diseño de sistemas de recomendación de libros.
            - Posibilidad de adaptar el enfoque a otros dominios textuales.
        
        ---
        
        ### **Modelos y comparación**
        
        1. **TF-IDF + LDA:**
            - Simplicidad y efectividad en modelado de temas, pero limitado en análisis emocional.
        2. **Bi-gramMSA con CNN:**
            - Supera a los métodos tradicionales en precisión y profundidad analítica.
            - Integra datos multimodales, permitiendo un análisis más rico.
        3. **Algoritmos supervisados vs. no supervisados:**
            - Los supervisados ofrecen mejores resultados en análisis de sentimientos, mientras que los no supervisados son útiles para descubrir temas latentes.
            
            ### **Algoritmos supervisados**
            
            - **Redes neuronales convolucionales (CNN):**
                - Utilizadas principalmente para el análisis de sentimientos a partir de bi-gramas y datos multimodales (texto, imágenes, audio).
                - Entrenadas con un corpus anotado manualmente para predecir la polaridad emocional.
            - **Support Vector Machines (SVM):**
                - Aplicadas como clasificador para tareas de polaridad emocional y categorización de temas literarios en un conjunto de clases predefinidas.
            - **Árboles de decisión:**
                - Usados para tomar decisiones basadas en características específicas extraídas de los textos, como la frecuencia de términos y patrones en los bi-gramas.
            
            ---
            
            ### **Algoritmos no supervisados**
            
            - **Latent Dirichlet Allocation (LDA):**
                - Método popular para modelado de temas. Se emplea para identificar temas latentes dentro de los textos literarios basándose en la coocurrencia de palabras.
            - **Clustering basado en k-means:**
                - Utilizado para agrupar textos similares en función de las características léxicas y contextuales extraídas.
            - **Análisis de componentes principales (PCA):**
                - Aplicado para reducir la dimensionalidad de las representaciones textuales y mejorar la eficiencia del clustering.
        
        ---
        
        ### **Características extraídas**
        
        - **Frecuencias léxicas:** A través de TF-IDF y bi-gramas.
        - **Embeddings textuales y visuales:** Generados por Word2Vec y CNNs, respectivamente.
        - **Stopwords específicas del dominio:** Identificadas para mejorar la representación textual.
        
        ---
        
        ### **Dataset**
        
        - **Corpus:** Obras literarias angloamericanas, incluyendo textos, ilustraciones y anotaciones de audio.
        - **Anotaciones:** Polaridad emocional y temas etiquetados manualmente por expertos.
        - **Tamaño:** Más de 500 libros representativos de distintos períodos y géneros.

- [https://www.tandfonline.com/doi/abs/10.1080/01639374.2024.2315548](https://www.tandfonline.com/doi/abs/10.1080/01639374.2024.2315548)
    - **Nombre:** Time Period Categorization in Fiction: A Comparative Analysis of Machine Learning Techniques
    - **Fecha**: 2024
    - **Resumen**:
        
        Este artículo analiza la categorización automática de textos literarios de ficción histórica en períodos de tiempo específicos mediante el uso de técnicas de aprendizaje automático. A continuación, se realiza un análisis detallado de las secciones clave.
        
        ---
        
        ### **Introducción**
        
        - **Objetivo:** Explorar cómo diferentes técnicas de procesamiento de lenguaje natural (PLN) y aprendizaje automático pueden clasificar fragmentos de ficción histórica en períodos de tiempo específicos.
        - **Motivación:** Las novelas históricas a menudo incluyen referencias a eventos, personajes y estilos lingüísticos que pueden indicar el período en el que están ambientadas. Automatizar esta tarea puede ayudar en estudios literarios y en sistemas de recomendación de libros.
        - **Aporte:** Primera comparación sistemática de varias técnicas de aprendizaje automático para categorizar períodos temporales en textos literarios.
        
        ---
        
        ### **Data Collection**
        
        - **Corpus:** Seleccionado de la **Swedish Literature Bank**, un repositorio digital de literatura sueca.
        - **Estructura:**
            - **35 novelas históricas** de distintos períodos.
            - Fragmentos divididos en aproximadamente **400 segmentos** de 3,500 palabras cada uno.
        - **Categorías temporales:** Cuatro períodos históricos principales:
            - Era Medieval.
            - Era de Gran Poder.
            - Período Gustaviano.
            - Edad de la Libertad.
        - **Etiquetado:** Los fragmentos fueron etiquetados manualmente según el período histórico predominante.
        
        ---
        
        ### **Feature Extraction**
        
        1. **TF-IDF (Term Frequency-Inverse Document Frequency):**
            - Representa los textos como vectores en los que cada palabra tiene un peso basado en su relevancia en el corpus.
            - Mejora la discriminación entre palabras comunes y términos específicos de un período histórico.
        2. **LDA (Latent Dirichlet Allocation):**
            - Modelo probabilístico que identifica temas latentes dentro de los textos.
            - Cada fragmento de texto recibe una distribución de probabilidades sobre los temas identificados.
        3. **SBERT (Sentence-BERT):**
            - Genera representaciones de alta dimensionalidad basadas en embeddings semánticos para medir similitudes entre fragmentos.
            - Aunque efectivo en otros contextos, su rendimiento fue limitado en este estudio debido a la naturaleza histórica del corpus.
        
        ---
        
        ### **Machine Learning Models**
        
        1. **Support Vector Machines (SVM):**
            - Utilizado por su capacidad para manejar espacios de alta dimensionalidad generados por TF-IDF.
            - Demostró ser uno de los modelos más efectivos.
        2. **Logistic Regression:**
            - Modelo lineal aplicado a las probabilidades generadas por LDA y TF-IDF.
        3. **Neural Networks (MLP):**
            - Redes de perceptrón multicapa capaces de capturar relaciones no lineales entre las características.
        4. **Random Forest:**
            - Modelo basado en árboles, aunque menos efectivo para esta tarea debido a la complejidad semántica del corpus.
        
        ---
        
        ### **Results**
        
        - **Comparación de técnicas:**
            - **TF-IDF:** La técnica más efectiva, con F1-scores de entre **0.92 y 0.99**.
            - **LDA:** Adecuado, con F1-scores de entre **0.74 y 0.92**.
            - **SBERT:** Puntuaciones más bajas, entre **0.39 y 0.52**, debido a la falta de ajuste a textos históricos específicos.
        - **Modelos:**
            - **SVM:** Mejor rendimiento general combinado con TF-IDF.
            - **MLP:** Similar a SVM, aunque más costoso computacionalmente.
            - **Random Forest:** Menor precisión comparado con SVM y MLP, pero más rápido.
        - **Impacto de las categorías:** Períodos con menos datos tuvieron puntuaciones más bajas, destacando la necesidad de datos balanceados.
        
        ---
        
        ### **Discussion**
        
        - **Hallazgos principales:**
            - TF-IDF es altamente efectivo para capturar características lingüísticas específicas de períodos históricos.
            - LDA es útil para identificar temas latentes, pero requiere más ajustes para lograr una precisión comparable.
            - SBERT, aunque prometedor, no es adecuado para dominios históricos sin ajustes específicos.
        - **Limitaciones:**
            - Corpus limitado en tamaño y diversidad.
            - Dependencia en etiquetas manuales, lo que introduce sesgos humanos.
        - **Implicaciones:**
            - Las técnicas estudiadas pueden extenderse a otros dominios, como análisis temáticos generales o categorización de documentos históricos.
        
        ---
        
        ### **Conclusion**
        
        - **Conclusión general:** TF-IDF combinado con SVM o redes neuronales ofrece el mejor rendimiento para la categorización de períodos históricos en textos literarios.
        - **Impacto:** Los resultados pueden aplicarse en análisis literario automatizado, sistemas de recomendación y archivística digital.
        - **Futuras líneas de trabajo:**
            - Ampliar el corpus a más períodos históricos y otros idiomas.
            - Adaptar SBERT mediante fine-tuning para textos históricos.
        
        ---
        
- https://ieeexplore.ieee.org/abstract/document/9115602
    - **Nombre**: Using Machine Learning and Thematic Analysis Methods to Evaluate Mental Health Apps Based on User Reviews
    - **Fecha**: 2023
    - **Resumen**:
        
        Este artículo combina técnicas de aprendizaje automático y análisis temático para evaluar aplicaciones de salud mental utilizando las opiniones de los usuarios. Su objetivo principal es identificar factores clave que afectan la eficacia de las aplicaciones, tanto positivos como negativos, para proporcionar recomendaciones de diseño basadas en datos.
        
        ---
        
        ### **Objetivo**
        
        Desarrollar un enfoque híbrido que utilice aprendizaje automático para clasificar las opiniones de los usuarios en términos de polaridad (positivo/negativo) y análisis temático para identificar temas recurrentes relacionados con la experiencia del usuario.
        
        ---
        
        ### **Metodología**
        
        1. **Recopilación de datos:**
            - Más de **88,125 opiniones** de usuarios de aplicaciones de salud mental obtenidas de **Google Play** y **App Store**.
            - **105 aplicaciones de salud mental** analizadas.
        2. **Preprocesamiento:**
            - Normalización textual, tokenización, eliminación de palabras irrelevantes (stopwords) y lematización.
        3. **Análisis supervisado (aprendizaje automático):**
            - **Modelos utilizados:**
                - Support Vector Machines (SVM).
                - Logistic Regression (LR).
                - Multinomial Naïve Bayes (MNB).
                - Random Forest (RF).
                - Stochastic Gradient Descent (SGD).
            - SGD fue el modelo con mejor desempeño, con un F1-score de **89.42%**.
        4. **Análisis temático:**
            - Identificación manual de temas recurrentes en las opiniones clasificadas, que se agruparon en categorías como:
                - **Aspectos positivos:** Personalización, interfaz atractiva, facilidad de uso.
                - **Aspectos negativos:** Problemas técnicos, costos altos, privacidad deficiente.
        
        ---
        
        ### **Resultados**
        
        - **Análisis de polaridad:**
            - Clasificación de opiniones en positivas y negativas con alta precisión (F1-score promedio superior al 85%).
        - **Temas identificados:**
            - **29 temas positivos:** Como la accesibilidad y el contenido interactivo.
            - **21 temas negativos:** Como la falta de soporte técnico y costos excesivos.
        - **Recomendaciones:** Mejorar aspectos como la privacidad del usuario, la usabilidad, y la inclusión de contenido clínicamente validado.
        
        ---
        
        ### **Impacto**
        
        El artículo proporciona una metodología efectiva para analizar grandes volúmenes de datos textuales, ayudando a diseñar aplicaciones de salud mental más eficaces. Las técnicas utilizadas son transferibles a otros dominios, como el análisis de satisfacción del cliente o la evaluación de productos digitales.
        
        ---
        
        ### **Conclusión**
        
        La combinación de aprendizaje automático y análisis temático permite evaluar eficazmente la calidad y efectividad de las aplicaciones de salud mental desde la perspectiva del usuario. El enfoque propuesto destaca por su capacidad para manejar grandes volúmenes de datos, identificar patrones clave y ofrecer recomendaciones prácticas para mejorar las aplicaciones.
        
- https://dl.acm.org/doi/abs/10.1145/3439726
    - **Nombre**: Deep Learning–based Text Classification: A Comprehensive Review
    - **Fecha**: 2021
    - **Resumen:** Survey