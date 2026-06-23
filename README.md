# Colorización Automática de Imágenes mediante una Arquitectura Híbrida

Este proyecto aborda el problema de la **colorización automática de imágenes en escala de grises**. A través de un enfoque híbrido que combina Redes Neuronales Convolucionales (CNN) tipo U-Net, Perceptrones Multicapa (MLP) y una capa personalizada de Funciones de Base Radial (RBF), el modelo aprende a reconstruir componentes cromaticos de imágenes basadas en caricaturas populares.

## Descripción del Proyecto

La colorización de imágenes es un desafío clásico y complejo en visión por computadora debido a su naturaleza ambigua (múltiples combinaciones de colores pueden ser válidas para una misma combinación de grises). 

Mientras que los enfoques tradicionales emplean únicamente convoluciones secuenciales (las cuales tienden a generar imágenes borrosas o colores promediados), este proyecto propone y evalúa una arquitectura de ensamble híbrida. El modelo extrae características geométricas mediante un bloque codificador-decodificador y diversifica la toma de decisiones utilizando dos aproximaciones matemáticas complementarias: una global lineal/no-lineal (MLP) y una basada en distancias y prototipos de color (RBF).

## Dataset

El proyecto utiliza el dataset público [Cartoon Classification](https://www.kaggle.com/datasets/volkandl/cartoon-classification) disponible en Kaggle.

* **Estructura:** El conjunto de datos original consta de **10 clases diferentes**, que corresponden a 10 personajes icónicos de caricaturas populares.
* **Volumen:** Está dividido balanceadamente con **10,000 imágenes por clase para el entrenamiento** (100k en total) y **2,000 imágenes por clase para pruebas** (20k en total).
* **Preprocesamiento en el Proyecto:** Para garantizar su viabilidad computacional en entornos de memoria limitada, el pipeline del código optimiza la carga de datos acotando el muestreo de manera segura, transformando las matrices a precisión reducida (`float16`) y vectorizando la conversión de color al espacio **CIE L\*a\*b*** una sola vez antes de alimentar la GPU.

## Arquitectura de la Red Neuronal (Ensamble Híbrido)

El flujo de información dentro del modelo desarrollado se divide en cuatro etapas fundamentales:

### 1. Codificador y Decodificador (Estructura U-Net)
* **Encoder (Contracción):** Se aplican bloques convolucionales iterativos con *strides* para extraer mapas de características abstractas de alta dimensionalidad semántica a partir del canal de luminancia.
* **Decoder (Expansión):** Se realiza un remuestreo (*UpSampling*) bilineal para recuperar las dimensiones espaciales originales de la imagen ($128 \times 128$ píxeles).
* **Skip Connections (Conexiones de Salto):** Se concatenan las salidas simétricas del codificador con el decodificador. Esto rescata la información geométrica fina (bordes nítidos y texturas) que normalmente se destruye en las redes convolucionales profundas tradicionales como AlexNet o LeNet-5.

### 2. Rama RBF (Capa Personalizada)
Se implementa una capa matemática RBF no convencional (`CapaRBFEspacial`). Esta capa proyecta el espacio latente del decodificador midiendo la distancia euclidiana de los píxeles respecto a **30 centros o prototipos de color aprendibles** en la imagen. Su activación matemática se rige bajo la función gaussiana:

$$\phi(x) = \exp(-\gamma ||x - \mu||^2)$$

Esto permite modelar regiones de color densas y continuas de manera muy eficiente.

### 3. Rama MLP
En paralelo a la RBF, un bloque de convoluciones $1 \times 1$ funciona como un Perceptrón Multicapa denso por píxel, encargándose de aprender las combinaciones globales de iluminación y combinaciones de color lineales.

### 4. Capa de Fusión Final
Las salidas de la rama MLP y la rama RBF se concatenan. Una última convolución aprende qué porcentaje de peso asignarle a cada cabeza para generar la predicción final de los canales cromáticos.
