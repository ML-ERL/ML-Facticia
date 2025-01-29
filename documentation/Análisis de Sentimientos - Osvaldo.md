# Análisis de sentimientos

El análisis de sentimientos es una técnica dentro del campo de la **procesamiento de lenguaje natural (NLP)** que se utiliza para determinar el tono, la emoción o la actitud expresada en un texto. Es decir, identifica si un fragmento de texto expresa un sentimiento positivo, negativo, neutro, o incluso más emociones complejas como alegría, enojo, sorpresa, etc.
Actualmente con la llegada de los LLMs (Large Language Models) como GPT-3, BERT, etc., se ha logrado un gran avance en el análisis de sentimientos, ya que estos modelos pueden capturar mejor el contexto y la semántica de las palabras en un texto.

---

### Herramientas actuales

- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: Es una herramienta de análisis de sentimientos basada en reglas y léxicos que se utiliza para analizar sentimientos en texto. VADER es especialmente útil cuando se trata de analizar sentimientos en redes sociales y otros textos cortos. [documentación](https://vadersentiment.readthedocs.io/en/latest/pages/installation.html)
  En la documentación ponen que descomponiendolos en textos pequeños se puede analizar un texto grande.

- **TextBlob**: Es una biblioteca de procesamiento de lenguaje natural (NLP) simple y fácil de usar para Python. Proporciona una API sencilla para sumergirse en tareas comunes de NLP como el análisis de sentimientos. [documentación](https://textblob.readthedocs.io/en/dev/). No hay muchos detalles de hasta qeu punto puede llegar (https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis)

- **BERT (Bidirectional Encoder Representations from Transformers)**: Es un modelo de lenguaje preentrenado desarrollado por Google que se puede utilizar para realizar una variedad de tareas de NLP, incluido el análisis de sentimientos. BERT es un modelo de lenguaje bidireccional que captura mejor el contexto y la semántica de las palabras en un texto. [documentación](https://huggingface.co/transformers/model_doc/bert.html). ROBERTA es una versión mejorada de BERT.(https://huggingface.co/docs/transformers/model_doc/roberta). Notebook de Fine-Tuning para felling (https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb)

- **GPT**: Es un modelo de lenguaje preentrenado desarrollado por OpenAI que se puede utilizar para realizar una variedad de tareas de NLP, incluido el análisis de sentimientos. GPT es un modelo de lenguaje generativo que puede generar texto coherente y relevante. [documentación](https://beta.openai.com/docs/)

- **DialogueLLM**: Es un modelo de lenguaje preentrenado desarrollado por Google que se puede utilizar para reconocer emociones en conversaciones. DialogueLLM es un modelo de lenguaje bidireccional que captura mejor el contexto y la semántica de las palabras en una conversación. [documentación](https://arxiv.org/abs/2310.11374) **(TODO: Buscar ek modelo)**

---

### Comparativa de ventajas y desventajas

| Modelo                          | Ventajas                                                                        | Desventajas                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| VADER                           | - Fácil de usar<br>- Rápido<br>- Bueno para textos cortos                       | - No captura matices culturales o contextuales<br>- Limitado en lenguajes no estándar |
| TextBlob                        | - Sencillo de implementar<br>- API intuitiva                                    | - Menor precisión que modelos avanzados<br>- Limitaciones en tareas complejas         |
| BERT                            | - Alta precisión<br>- Captura contexto y semántica                              | - Requiere hardware potente<br>- Entrenamiento largo y costoso                        |
| ROBERTA                         | - Mejora sobre BERT<br>- Más precisa                                            | - Mismo problema de hardware que BERT                                                 |
| GPT                             | - Capacidad generativa<br>- Buena para tareas secuenciales                      | - Difícil de evaluar objetivamente<br>- Sesgos potenciales                            |
| DialogueLLM                     | - Optimizado para conversaciones<br>- Captura emociones en contexto             | - Modelo reciente, menos documentación disponible                                     |
| CNN para detección de emociones | - Bueno para imágenes faciales<br>- Capacidad para detectar múltiples emociones | - Menor precisión en textos<br>- Requiere grandes conjuntos de datos etiquetados      |

### Conclusiones:

1. Para análisis de sentimientos básicos, VADER y TextBlob son buenas opciones por su facilidad de uso.
2. Para tareas más complejas, BERT, ROBERTA o GPT ofrecen mejor precisión pero requieren recursos computacionales más intensivos.
3. DialogueLLM parece prometedor para análisis de emociones en conversaciones.
4. CNNs pueden ser útiles para detección de emociones en imágenes faciales.

### Tipos de análisis de sentimientos

1. **Binario**: Determinar si el sentimiento es positivo o negativo.
2. **Multiclase**: Clasificar en más categorías como positivo, negativo y neutro.
3. **Basado en emociones**: Detectar emociones específicas como felicidad, tristeza, enojo, etc.
4. **Análisis aspectual**: Identificar el sentimiento sobre aspectos específicos dentro de un texto (por ejemplo, "la comida estaba deliciosa, pero el servicio fue lento").

## Links Importantes

- [https://www.elastic.co/es/what-is/sentiment-analysis]
- https://www.geeksforgeeks.org/dataset-for-sentiment-analysis/ (**Datasets**)
- https://medium.com/@pakhila413/emotion-detection-using-cnn-exploring-kaggle-dataset-and-model-architectures-c74fd34aebbf (**CNN**)
- https://www.sciencedirect.com/science/article/pii/S2949719124000074 (**Review del estado del arte**)
- https://arxiv.org/abs/2310.11374 (**DialogueLLM para reconocimiento de emociones en conversaciones**)
- https://www.nltk.org/

## Otros:

[1] https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair
[2] https://www.analyticsvidhya.com/blog/2021/01/sentiment-analysis-vader-or-textblob/
[3] https://medium.com/@hhpatil001/textblob-vs-vader-for-sentiment-analysis-9d36b0b79ae6
[4] https://www.analyticsvidhya.com/blog/2021/10/sentiment-analysis-with-textblob-and-vader/
[5] https://aashishmehta.com/sentiment-analysis-comparison/
[6] https://www.sciencedirect.com/science/article/pii/S2949719124000074
[7] https://arxiv.org/pdf/2307.14311
[8] https://www.kaggle.com/code/buyuknacar/comparing-sentiment-classifiers-txblob-vader-bert
[9] https://towardsdatascience.com/comparing-vader-and-text-blob-to-human-sentiment-77068cf73982
[10] https://arxiv.org/html/2401.08508v2
