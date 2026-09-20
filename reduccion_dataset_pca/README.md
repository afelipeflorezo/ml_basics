# 📉 Reducción Dimensional con PCA (Análisis de Componentes Principales)

Técnica de aprendizaje no supervisado que reduce el número de variables de un dataset manteniendo la mayor cantidad de información posible, usando Análisis de Componentes Principales (PCA).

---

## 📁 Estructura del Módulo

```
reduccion_dataset_pca/
├── README.md                       ← Este archivo
├── reduccion_dataset_pca.ipynb     ← Notebook con el pipeline completo
├── dataset_tabular.csv             ← Dataset original (4 variables + etiqueta)
├── dataset_reducido_pca.csv        ← Dataset reducido (2 componentes + etiqueta)
└── modelo_pca.pkl                  ← Modelo PCA entrenado
```

---

## 📊 Dataset: `dataset_tabular.csv`

Basado en el dataset **Iris** exportado a CSV con nombres en español.

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `id` | int | Identificador de la muestra |
| `largo_sepalo` | float | Largo del sépalo (cm) |
| `ancho_sepalo` | float | Ancho del sépalo (cm) |
| `largo_petalo` | float | Largo del pétalo (cm) |
| `ancho_petalo` | float | Ancho del pétalo (cm) |
| `etiqueta` | int | Especie (0, 1, 2) |

- **Total de muestras:** 150 registros
- **Dimensionalidad original:** 4 features → **Reducido a 2 componentes**

---

## 🧮 Fundamento Teórico

### Análisis de Componentes Principales (PCA)

PCA transforma un conjunto de variables posiblemente correlacionadas en un nuevo conjunto de variables **no correlacionadas** llamadas **componentes principales**, ordenadas por la cantidad de varianza que capturan:

1. **Estandarización:** Se escalan las variables a media=0, desviación=1.
2. **Cálculo de la matriz de covarianza.**
3. **Descomposición en autovalores y autovectores.**
4. **Selección de componentes:** Se eligen los K componentes que capturan suficiente varianza.
5. **Proyección:** Se transforman los datos al nuevo espacio de menor dimensión.

```
4 variables originales → PCA → 2 componentes principales
(que capturan ~95% de la varianza total)
```

> **Ventaja:** Reduce ruido, mejora visualización, acelera modelos posteriores.  
> **Nota:** Los componentes resultantes no tienen interpretación directa como las variables originales.

---

## ⚙️ Pipeline de Procesamiento

1. **Carga de datos:** Dataset Iris desde Scikit-Learn (también disponible en `dataset_tabular.csv`).
2. **Escalamiento:** `StandardScaler` para normalizar las 4 features.
3. **PCA completo:** `PCA()` sin restricción para calcular la varianza explicada de todos los componentes.
4. **Análisis de varianza:** Gráfico de varianza acumulada para elegir K.
5. **PCA reducido:** `PCA(n_components=2)` para proyectar a 2 dimensiones.
6. **Visualización:** Scatter plot 2D coloreado por especie.
7. **Serialización:** `joblib.dump(pca, 'modelo_pca.pkl')`
8. **Exportación:** `dataset_reducido_pca.csv` con las 2 componentes.

---

## 📈 Métricas y Resultados

| Métrica | Descripción |
|---------|-------------|
| **Varianza Explicada por Componente** | Proporción de la varianza total capturada por cada componente |
| **Varianza Acumulada** | Total acumulado — al llegar a ~95% se considera suficiente |

### Resultado: `dataset_reducido_pca.csv`

| id | componente_1 | componente_2 | etiqueta |
|----|-------------|-------------|----------|
| 0 | -2.26... | 0.48... | 0 |
| 1 | -2.08... | -0.67... | 0 |

---

## 🔮 Cómo Usar el Modelo Entrenado

```python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar el modelo PCA
pca = joblib.load("modelo_pca.pkl")

# Nuevos datos (4 features originales)
nuevos_datos = np.array([[5.1, 3.5, 1.4, 0.2]])

# IMPORTANTE: escalar antes de transformar
# (usar el mismo escalador del entrenamiento, o re-escalar con los mismos parámetros)
escalador = StandardScaler()
# En producción, guardar y cargar el escalador junto con PCA

datos_reducidos = pca.transform(nuevos_datos)
print(f"Componentes: {datos_reducidos}")
```

---

## ▶️ Ejecución

```bash
cd reduccion_dataset_pca
jupyter notebook reduccion_dataset_pca.ipynb
```

---

[⬅️ Volver al índice principal](../README.md)
