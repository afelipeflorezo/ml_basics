# 🌸 Clasificador Iris con K-Nearest Neighbors (KNN)

Clasificador supervisado que identifica la especie de una flor Iris a partir de las dimensiones de sus sépalos y pétalos, usando el algoritmo K-Nearest Neighbors.

---

## 📁 Estructura del Módulo

```
clasificador_iris_knn/
├── README.md                        ← Este archivo
├── clasificador_iris_knn.ipynb      ← Notebook con el pipeline completo
└── modelo_iris_knn.pkl              ← Modelo KNN entrenado (serializado)
```

---

## 📊 Dataset

Se utiliza el **dataset Iris** integrado en Scikit-Learn (`sklearn.datasets.load_iris`).

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `sepal length (cm)` | float | Largo del sépalo |
| `sepal width (cm)` | float | Ancho del sépalo |
| `petal length (cm)` | float | Largo del pétalo |
| `petal width (cm)` | float | Ancho del pétalo |
| **Target** | int | 0 = Setosa, 1 = Versicolor, 2 = Virginica |

- **Total de muestras:** 150 (50 por clase)
- **Clases:** 3 especies perfectamente balanceadas

---

## 🧮 Fundamento Teórico

### K-Nearest Neighbors (KNN)

KNN es un algoritmo de clasificación basado en instancias (*instance-based learning*). Para clasificar un nuevo punto:

1. Calcula la **distancia** (euclídea) entre el punto nuevo y todos los puntos de entrenamiento.
2. Selecciona los **K vecinos más cercanos**.
3. Asigna la **clase mayoritaria** entre esos K vecinos.

> **Ventaja:** Simple e intuitivo, no requiere fase de entrenamiento explícita.  
> **Desventaja:** Sensible a la escala de las variables → se usa `StandardScaler`.

---

## ⚙️ Pipeline de Procesamiento

1. **Carga de datos:** Dataset Iris desde Scikit-Learn.
2. **División:** 80% entrenamiento / 20% prueba (`train_test_split`, `random_state=42`).
3. **Modelo Pipeline:**
   - `StandardScaler` → Normaliza las 4 features a media=0, desviación=1.
   - `KNeighborsClassifier(n_neighbors=5)` → Clasificación por mayoría de 5 vecinos.
4. **Entrenamiento:** `modelo.fit(X_entrenamiento, y_entrenamiento)`
5. **Evaluación:** Accuracy, Classification Report y Confusion Matrix.
6. **Serialización:** `joblib.dump(modelo, 'modelo_iris_knn.pkl')`

---

## 📈 Métricas y Resultados

| Métrica | Valor |
|---------|-------|
| **Accuracy** | ~1.00 (100%) |

El dataset Iris es relativamente simple y las 3 clases son linealmente separables (especialmente Setosa), lo que permite a KNN con K=5 alcanzar precisión perfecta o cercana al 100%.

---

## 🔮 Cómo Usar el Modelo Entrenado

```python
import joblib

# Cargar el modelo (incluye StandardScaler + KNN)
modelo = joblib.load("modelo_iris_knn.pkl")

# Nuevas mediciones: [largo_sepalo, ancho_sepalo, largo_petalo, ancho_petalo]
nueva_flor = [[5.1, 3.5, 1.4, 0.2]]

prediccion = modelo.predict(nueva_flor)
clases = ["Setosa", "Versicolor", "Virginica"]
print(f"Especie predicha: {clases[prediccion[0]]}")
```

---

## ▶️ Ejecución

```bash
cd clasificador_iris_knn
jupyter notebook clasificador_iris_knn.ipynb
```

---

[⬅️ Volver al índice principal](../README.md)
