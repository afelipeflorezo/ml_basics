# 🔍 Detección de Fraude con Árboles de Decisión

Clasificador supervisado que identifica transacciones fraudulentas a partir de patrones en el monto, hora, ubicación y comportamiento del usuario, usando un Árbol de Decisión.

---

## 📁 Estructura del Módulo

```
fraud_detection/
├── README.md                           ← Este archivo
├── deteccion_fraude_arbol.ipynb        ← Notebook con el pipeline completo
├── transacciones.csv                   ← Dataset de transacciones
└── modelo_deteccion_fraude.pkl         ← Modelo Decision Tree entrenado
```

---

## 📊 Dataset: `transacciones.csv`

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `transaccion_id` | int | Identificador único de la transacción |
| `monto` | float | Monto de la transacción |
| `hora` | int | Hora del día (0-23) |
| `distancia_ubicacion_km` | float | Distancia al lugar habitual (km) |
| `cantidad_transacciones_dia` | int | Número de transacciones ese día |
| `es_comercio_nuevo` | int | 1 = comercio no visitado antes, 0 = conocido |
| `dispositivo_nuevo` | int | 1 = dispositivo nuevo, 0 = habitual |
| **`fraude`** | int | **Variable objetivo** — 1 = fraude, 0 = legítima |

- **Total de muestras:** 1,000 transacciones

---

## 🧮 Fundamento Teórico

### Árbol de Decisión (Decision Tree)

Un árbol de decisión divide recursivamente el espacio de features en regiones, eligiendo en cada nodo la **variable y el umbral** que mejor separan las clases (minimizando el índice Gini o la entropía).

```
               ¿monto > 500?
              /              \
         Sí                    No
     ¿dispositivo_nuevo?     → Legítima
    /                  \
  Sí                    No
→ Fraude             → Legítima
```

> **Ventaja:** Altamente interpretable, se puede visualizar como un diagrama de flujo.  
> **Desventaja:** Propenso a sobreajuste si no se controla la profundidad.

---

## ⚙️ Pipeline de Procesamiento

1. **Carga de datos:** `transacciones.csv` con pandas.
2. **Separación de variables:** X = features (sin `transaccion_id` ni `fraude`), y = `fraude`.
3. **División:** 80% entrenamiento / 20% prueba (`random_state=42`).
4. **Entrenamiento:** `DecisionTreeClassifier().fit(X_train, y_train)`
5. **Evaluación:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
6. **Visualización:** Árbol de decisión graficado con `plot_tree`.
7. **Serialización:** `joblib.dump(modelo, 'modelo_deteccion_fraude.pkl')`

---

## 📈 Métricas y Resultados

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | Proporción total de predicciones correctas |
| **Precision** | De las alertadas como fraude, ¿cuántas realmente lo son? |
| **Recall** | De los fraudes reales, ¿cuántos fueron detectados? |
| **F1-Score** | Media armónica de Precision y Recall |

> En detección de fraude, **Recall** es la métrica más crítica: es preferible generar falsas alarmas a dejar pasar un fraude real.

---

## 🔮 Cómo Usar el Modelo Entrenado

```python
import joblib

# Cargar el modelo
modelo = joblib.load("modelo_deteccion_fraude.pkl")

# Nueva transacción: [monto, hora, distancia_km, transacciones_dia, comercio_nuevo, dispositivo_nuevo]
nueva_transaccion = [[1500.0, 3, 85.0, 8, 1, 1]]

prediccion = modelo.predict(nueva_transaccion)
etiqueta = "🚨 FRAUDE" if prediccion[0] == 1 else "✅ Legítima"
print(f"Resultado: {etiqueta}")
```

---

## ▶️ Ejecución

```bash
cd fraud_detection
jupyter notebook deteccion_fraude_arbol.ipynb
```

---

[⬅️ Volver al índice principal](../README.md)
