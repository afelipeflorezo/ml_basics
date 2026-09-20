# Guía Práctica de Machine Learning en Python

Repositorio educativo con **10 proyectos prácticos** de Machine Learning clásico y Deep Learning, diseñado como referencia de aprendizaje progresivo. Cada proyecto es autocontenido e incluye dataset, notebook interactivo y modelo pre-entrenado listo para usar.

---

## 📋 Tabla de Contenidos

- [Taxonomía de Proyectos](#-taxonomía-de-proyectos)
- [Hoja de Ruta de Aprendizaje](#-hoja-de-ruta-de-aprendizaje)
- [Mapa del Repositorio](#-mapa-del-repositorio)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Guía Rápida de Inferencia](#-guía-rápida-de-inferencia)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)

---

## 🗂 Taxonomía de Proyectos

| # | Proyecto | Paradigma | Algoritmo | Métricas Clave |
|:-:|----------|-----------|-----------|----------------|
| 1 | [Clasificador Iris (KNN)](./clasificador_iris_knn/) | Supervisado · Clasificación | K-Nearest Neighbors | Accuracy, Confusion Matrix |
| 2 | [Predicción Precio Casas](./house_price_prediction/) | Supervisado · Regresión | Regresión Lineal Multivariable | R², MAE, RMSE |
| 3 | [Detección de Fraude](./fraud_detection/) | Supervisado · Clasificación | Árbol de Decisión | Accuracy, Precision, Recall, F1 |
| 4 | [Clasificador Spam (Naive Bayes)](./naive-bayes-email-spam-classification/) | Supervisado · NLP | Naive Bayes Multinomial + TF-IDF | Accuracy, Classification Report |
| 5 | [Análisis de Sentimientos](./analisis_sentimientos/) | Supervisado · NLP | Regresión Logística + TF-IDF | Accuracy, Classification Report |
| 6 | [Clustering de Clientes](./clustering_clientes_kmeans/) | No Supervisado · Clustering | K-Means (Elbow + Silhouette) | Silhouette Score, Inercia |
| 7 | [Reducción PCA](./reduccion_dataset_pca/) | No Supervisado · Dimensionalidad | Análisis de Componentes Principales | Varianza Explicada |
| 8 | [Sistema de Recomendación](./sistema_recomendacion/) | Filtrado Colaborativo | Similitud Coseno | Top-N Recomendaciones |
| 9 | [Series Temporales de Ventas](./time_series_ventas/) | Series Temporales | Regresión con Rezagos (Lags) | R², MAE, RMSE |
| 10 | [MNIST (MLP vs CNN)](./mnist/) | Deep Learning · Visión | MLP y CNN con TensorFlow/Keras | Accuracy, Loss |

---

## 🗺 Hoja de Ruta de Aprendizaje

Orden sugerido para recorrer los proyectos, desde lo más intuitivo hasta técnicas avanzadas:

```
 NIVEL 1 — Fundamentos
 ├── 1. clasificador_iris_knn         → Tu primer clasificador (KNN + Iris)
 ├── 2. house_price_prediction        → Regresión lineal, métricas de error
 └── 3. reduccion_dataset_pca         → Entender PCA y reducción dimensional

 NIVEL 2 — Clasificación Avanzada
 ├── 4. fraud_detection               → Árboles de decisión, datos desbalanceados
 ├── 5. naive-bayes-email-spam        → NLP básico: TF-IDF + Naive Bayes
 └── 6. analisis_sentimientos         → NLP: pipeline TF-IDF + Regresión Logística

 NIVEL 3 — Aprendizaje No Supervisado
 ├── 7. clustering_clientes_kmeans    → Segmentación con K-Means
 └── 8. sistema_recomendacion         → Filtrado colaborativo con coseno

 NIVEL 4 — Modelos Temporales y Deep Learning
 ├── 9. time_series_ventas            → Pronóstico con features de rezago
 └── 10. mnist                        → Redes neuronales: MLP vs CNN (TensorFlow)
```

---

## 📁 Mapa del Repositorio

```
ml_basics/
│
├── README.md                              ← Este archivo
├── requirements.txt                       ← Dependencias globales
├── .gitignore                             ← Archivos ignorados por Git
│
├── clasificador_iris_knn/                 ← KNN sobre dataset Iris
│   ├── README.md
│   ├── clasificador_iris_knn.ipynb
│   └── modelo_iris_knn.pkl
│
├── house_price_prediction/                ← Regresión lineal de precios
│   ├── README.md
│   ├── house_price_prediction.ipynb
│   ├── casas.csv
│   ├── resultados_predicciones.csv
│   └── modelo_precios_casas.pkl
│
├── fraud_detection/                       ← Detección de fraude con árboles
│   ├── README.md
│   ├── deteccion_fraude_arbol.ipynb
│   ├── transacciones.csv
│   └── modelo_deteccion_fraude.pkl
│
├── naive-bayes-email-spam-classification/ ← Clasificador de spam
│   ├── README.md
│   ├── naive_bayes.ipynb
│   ├── correos.csv
│   └── modelo_detector_spam.pkl
│
├── analisis_sentimientos/                 ← Análisis de sentimientos NLP
│   ├── README.md
│   ├── analisis_sentimientos.ipynb
│   ├── reseñas.csv
│   ├── resultados_sentimientos.csv
│   └── modelo_sentimientos.pkl
│
├── clustering_clientes_kmeans/            ← Segmentación de clientes
│   ├── README.md
│   ├── clustering_clientes_kmeans.ipynb
│   ├── clientes.csv
│   ├── clientes_clasificados.csv
│   ├── perfil_grupos_clientes.csv
│   └── modelo_clientes_kmeans.pkl
│
├── reduccion_dataset_pca/                 ← Reducción dimensional con PCA
│   ├── README.md
│   ├── reduccion_dataset_pca.ipynb
│   ├── dataset_tabular.csv
│   ├── dataset_reducido_pca.csv
│   └── modelo_pca.pkl
│
├── sistema_recomendacion/                 ← Recomendador colaborativo
│   ├── README.md
│   ├── sistema_recomendacion.ipynb
│   ├── calificaciones.csv
│   └── modelo_recomendador.pkl
│
├── time_series_ventas/                    ← Pronóstico de ventas
│   ├── README.md
│   ├── time_series_ventas.ipynb
│   ├── serie_temporal.csv
│   ├── predicciones_futuras.csv
│   ├── resultados_series_temporales.csv
│   └── modelo_series_temporales.pkl
│
└── mnist/                                 ← Deep Learning: MLP vs CNN
    ├── README.md
    ├── mnist_mlp.ipynb
    ├── mnist_cnn.ipynb
    └── mnist_mlp_cnn.ipynb
```

---

## ⚙️ Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/afelipeflorezo/ml_basics.git
cd ml_basics
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Abrir los notebooks

```bash
jupyter notebook
```

Navega al directorio del proyecto que desees explorar y abre el archivo `.ipynb`.

---

## 🚀 Guía Rápida de Inferencia

Todos los modelos están serializados con `joblib`. Para cargar y usar cualquier modelo pre-entrenado:

```python
import joblib

# Cargar el modelo (ejemplo: clasificador Iris)
modelo = joblib.load("clasificador_iris_knn/modelo_iris_knn.pkl")

# Hacer una predicción
# (los datos de entrada deben tener el mismo formato que los de entrenamiento)
prediccion = modelo.predict([[5.1, 3.5, 1.4, 0.2]])
print(f"Clase predicha: {prediccion[0]}")
```

> **Nota:** Cada `README.md` de módulo incluye un snippet específico con el formato exacto de entrada para ese modelo.

---

## 🛠 Tecnologías Utilizadas

| Categoría | Tecnología |
|-----------|------------|
| Lenguaje | Python 3.8+ |
| ML Clásico | Scikit-Learn |
| Deep Learning | TensorFlow / Keras |
| Datos | Pandas, NumPy |
| Visualización | Matplotlib, Seaborn |
| Serialización | Joblib |
| Entorno | Jupyter Notebook |

---

## 📄 Licencia

Este repositorio es de uso educativo y de referencia personal.

---

<p align="center">
  <i>Desarrollado como guía de aprendizaje de Machine Learning por <a href="https://github.com/afelipeflorezo">@afelipeflorezo</a></i>
</p>
