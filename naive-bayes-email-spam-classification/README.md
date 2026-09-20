# 📧 Clasificador de Spam con Naive Bayes

Clasificador supervisado de correos electrónicos que distingue entre mensajes legítimos (ham) y spam, usando Naive Bayes Multinomial con vectorización TF-IDF.

---

## 📁 Estructura del Módulo

```
naive-bayes-email-spam-classification/
├── README.md                   ← Este archivo
├── naive_bayes.ipynb           ← Notebook con el pipeline completo
├── correos.csv                 ← Dataset de correos etiquetados
└── modelo_detector_spam.pkl    ← Pipeline TF-IDF + Naive Bayes entrenado
```

---

## 📊 Dataset: `correos.csv`

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `texto` | string | Contenido del correo electrónico |
| **`etiqueta`** | string | **Variable objetivo** — `"ham"` (legítimo) o `"spam"` |

- **Total de muestras:** 20 correos
- **Distribución:** Correos de ejemplo que incluyen ofertas, premios falsos, mensajes personales, etc.

---

## 🧮 Fundamento Teórico

### Naive Bayes Multinomial

Aplica el **Teorema de Bayes** bajo la suposición de independencia condicional entre las palabras:

```
P(spam | texto) ∝ P(texto | spam) × P(spam)
```

- **TF-IDF (Term Frequency - Inverse Document Frequency):** Transforma el texto en un vector numérico donde cada palabra tiene un peso proporcional a su frecuencia en el documento e inversamente proporcional a su frecuencia global en el corpus.
- **Naive Bayes Multinomial:** Ideal para datos de conteo/frecuencia de texto. Calcula la probabilidad posterior de cada clase y asigna la más probable.

> **Ventaja:** Rápido, eficiente con texto, buen baseline para NLP.  
> **Desventaja:** La suposición de independencia rara vez se cumple en el lenguaje natural.

---

## ⚙️ Pipeline de Procesamiento

1. **Carga de datos:** `correos.csv` con pandas.
2. **División:** Train/Test split (`random_state=42`).
3. **Pipeline de Scikit-Learn:**
   - `TfidfVectorizer()` → Convierte texto a vectores TF-IDF.
   - `MultinomialNB()` → Clasificador probabilístico.
4. **Entrenamiento:** `pipeline.fit(X_train, y_train)`
5. **Evaluación:** Accuracy, Classification Report, Confusion Matrix.
6. **Serialización:** `joblib.dump(pipeline, 'modelo_detector_spam.pkl')`

---

## 📈 Métricas y Resultados

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | Proporción de correos correctamente clasificados |
| **Precision** | De los marcados como spam, ¿cuántos realmente lo son? |
| **Recall** | De los spam reales, ¿cuántos fueron detectados? |

---

## 🔮 Cómo Usar el Modelo Entrenado

```python
import joblib

# Cargar pipeline completo (TF-IDF + Naive Bayes)
modelo = joblib.load("modelo_detector_spam.pkl")

# Clasificar nuevos correos
nuevos_correos = [
    "¡Felicidades! Has ganado un iPhone. Haz clic aquí para reclamar.",
    "Hola, ¿nos vemos mañana para el almuerzo?"
]

predicciones = modelo.predict(nuevos_correos)
for correo, pred in zip(nuevos_correos, predicciones):
    icono = "🚫 SPAM" if pred == "spam" else "✅ Legítimo"
    print(f"{icono}: {correo[:50]}...")
```

---

## ▶️ Ejecución

```bash
cd naive-bayes-email-spam-classification
jupyter notebook naive_bayes.ipynb
```

---

[⬅️ Volver al índice principal](../README.md)
