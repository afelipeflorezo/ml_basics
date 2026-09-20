# 💬 Análisis de Sentimientos con Regresión Logística

Modelo de NLP supervisado que clasifica reseñas de texto como **positivas** o **negativas**, usando un pipeline de TF-IDF + Regresión Logística.

---

## 📁 Estructura del Módulo

```
analisis_sentimientos/
├── README.md                          ← Este archivo
├── analisis_sentimientos.ipynb        ← Notebook con el pipeline completo
├── reseñas.csv                        ← Dataset de reseñas etiquetadas
├── resultados_sentimientos.csv        ← Predicciones vs valores reales
└── modelo_sentimientos.pkl            ← Pipeline TF-IDF + LogReg entrenado
```

---

## 📊 Dataset: `reseñas.csv`

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `reseña` | string | Texto de la reseña del usuario |
| **`sentimiento`** | string | **Variable objetivo** — `"positivo"` o `"negativo"` |

- **Total de muestras:** 25 reseñas
- **Salida:** `resultados_sentimientos.csv` con 7 predicciones comparadas con el valor real.

---

## 🧮 Fundamento Teórico

### Pipeline TF-IDF + Regresión Logística

1. **TF-IDF Vectorizer:** Convierte cada reseña de texto en un vector numérico de alta dimensionalidad, ponderando cada término por su relevancia relativa al corpus completo.

2. **Regresión Logística:** Modelo lineal que estima la probabilidad de pertenencia a cada clase mediante la función sigmoide:

```
P(positivo | x) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))
```

> **Ventaja sobre Naive Bayes:** No asume independencia entre palabras; captura interacciones lineales.  
> **Uso típico:** Análisis de reseñas, encuestas de satisfacción, monitoreo de redes sociales.

---

## ⚙️ Pipeline de Procesamiento

1. **Carga de datos:** `reseñas.csv` con pandas.
2. **División:** Train/Test split (`random_state=42`).
3. **Pipeline de Scikit-Learn:**
   - `TfidfVectorizer()` → Vectorización de texto.
   - `LogisticRegression()` → Clasificación binaria.
4. **Entrenamiento:** `pipeline.fit(X_train, y_train)`
5. **Evaluación:** Accuracy, Classification Report, Confusion Matrix.
6. **Serialización:** `joblib.dump(pipeline, 'modelo_sentimientos.pkl')`
7. **Exportación:** Predicciones en `resultados_sentimientos.csv`.

---

## 📈 Métricas y Resultados

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | Proporción de reseñas correctamente clasificadas |
| **Classification Report** | Precision, Recall y F1 por clase (positivo/negativo) |

---

## 🔮 Cómo Usar el Modelo Entrenado

```python
import joblib

# Cargar pipeline completo (TF-IDF + Regresión Logística)
modelo = joblib.load("modelo_sentimientos.pkl")

# Clasificar nuevas reseñas
nuevas_reseñas = [
    "El producto es excelente, superó mis expectativas",
    "Pésimo servicio, no lo recomiendo para nada"
]

predicciones = modelo.predict(nuevas_reseñas)
for reseña, pred in zip(nuevas_reseñas, predicciones):
    emoji = "😊" if pred == "positivo" else "😞"
    print(f"{emoji} [{pred.upper()}] {reseña}")
```

---

## ▶️ Ejecución

```bash
cd analisis_sentimientos
jupyter notebook analisis_sentimientos.ipynb
```

---

[⬅️ Volver al índice principal](../README.md)
