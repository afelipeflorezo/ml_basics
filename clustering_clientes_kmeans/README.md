# 👥 Segmentación de Clientes con K-Means Clustering

Modelo de aprendizaje no supervisado que agrupa clientes en segmentos homogéneos según su edad, ingresos y puntuación de gasto, usando K-Means con validación por método del Codo y Silhouette Score.

---

## 📁 Estructura del Módulo

```
clustering_clientes_kmeans/
├── README.md                              ← Este archivo
├── clustering_clientes_kmeans.ipynb       ← Notebook con el pipeline completo
├── clientes.csv                           ← Dataset original de clientes
├── clientes_clasificados.csv              ← Clientes con grupo asignado
├── perfil_grupos_clientes.csv             ← Perfil promedio de cada grupo
└── modelo_clientes_kmeans.pkl             ← Modelo KMeans entrenado
```

---

## 📊 Dataset: `clientes.csv`

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `cliente_id` | int | Identificador único del cliente |
| `edad` | int | Edad del cliente |
| `ingresos_mensuales` | int | Ingresos mensuales (unidades monetarias) |
| `puntuacion_gasto` | int | Puntuación de gasto (1-100) |

- **Total de muestras:** 30 clientes
- **No hay variable objetivo** — el modelo descubre los grupos por sí mismo.

---

## 🧮 Fundamento Teórico

### K-Means Clustering

K-Means particiona N observaciones en K clusters, minimizando la **inercia** (suma de distancias cuadráticas al centroide más cercano):

1. Inicializa K centroides aleatoriamente.
2. **Asignación:** Cada punto se asigna al centroide más cercano.
3. **Actualización:** Cada centroide se recalcula como la media de sus puntos.
4. Repite 2-3 hasta convergencia.

### Selección del K óptimo

| Método | Criterio |
|--------|----------|
| **Método del Codo (Elbow)** | Busca el punto donde la inercia deja de decrecer significativamente |
| **Silhouette Score** | Mide qué tan bien cada punto encaja en su cluster vs el más cercano (-1 a 1) |

---

## ⚙️ Pipeline de Procesamiento

1. **Carga de datos:** `clientes.csv` con pandas.
2. **Selección de features:** `edad`, `ingresos_mensuales`, `puntuacion_gasto`.
3. **Escalamiento:** `StandardScaler` para normalizar las variables.
4. **Selección de K:** Método del Codo (inercia) + Silhouette Score para K = 2..10.
5. **Entrenamiento:** `KMeans(n_clusters=K_optimo, random_state=42).fit_predict(X_escalado)`
6. **Asignación de grupos:** Se añade columna `grupo` al dataset original.
7. **Perfilamiento:** Se calcula el perfil promedio de cada cluster.
8. **Serialización:** `joblib.dump(modelo, 'modelo_clientes_kmeans.pkl')`

---

## 📈 Métricas y Resultados

| Métrica | Descripción |
|---------|-------------|
| **Inercia** | Suma de distancias al centroide más cercano (menor = mejor) |
| **Silhouette Score** | Calidad de la separación entre clusters (más cercano a 1 = mejor) |

### Perfiles de Grupo: `perfil_grupos_clientes.csv`

Cada fila representa las características promedio de un grupo de clientes, permitiendo estrategias de marketing diferenciadas.

---

## 🔮 Cómo Usar el Modelo Entrenado

```python
import joblib
import numpy as np

# Cargar modelo y escalador
datos_modelo = joblib.load("modelo_clientes_kmeans.pkl")
# El archivo contiene el modelo KMeans y el escalador

# Nuevo cliente: [edad, ingresos_mensuales, puntuacion_gasto]
nuevo_cliente = np.array([[35, 45000, 72]])

# Escalar y predecir
# (Verificar la estructura del pkl para el escalador)
grupo = datos_modelo.predict(nuevo_cliente)
print(f"El cliente pertenece al grupo: {grupo[0]}")
```

---

## ▶️ Ejecución

```bash
cd clustering_clientes_kmeans
jupyter notebook clustering_clientes_kmeans.ipynb
```

---

[⬅️ Volver al índice principal](../README.md)
