# 🏠 Predicción de Precios de Casas — Regresión Lineal Multivariable

Modelo de regresión supervisada que predice el precio de una vivienda basándose en sus características físicas (área, habitaciones, baños, antigüedad y garajes).

---

## 📁 Estructura del Módulo

```
house_price_prediction/
├── README.md                        ← Este archivo
├── house_price_prediction.ipynb     ← Notebook con el pipeline completo
├── casas.csv                        ← Dataset de entrenamiento
├── resultados_predicciones.csv      ← Resultados de predicciones en test
└── modelo_precios_casas.pkl         ← Modelo de regresión entrenado
```

---

## 📊 Dataset: `casas.csv`

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `area_m2` | int | Área construida en metros cuadrados |
| `habitaciones` | int | Número de habitaciones |
| `banos` | int | Número de baños |
| `antiguedad` | int | Años de antigüedad de la vivienda |
| `garajes` | int | Número de garajes |
| **`precio`** | int | **Variable objetivo** — Precio de venta |

- **Total de muestras:** 5 registros
- **Nota:** Es un dataset educativo pequeño, diseñado para ilustrar el concepto.

---

## 🧮 Fundamento Teórico

### Regresión Lineal Multivariable

El modelo ajusta una función lineal de la forma:

```
precio = β₀ + β₁·area_m2 + β₂·habitaciones + β₃·baños + β₄·antigüedad + β₅·garajes
```

Donde los coeficientes βᵢ se calculan minimizando el **error cuadrático medio (MSE)** usando el método de mínimos cuadrados ordinarios (OLS).

> **Ventaja:** Interpretable, rápido, base para modelos más complejos.  
> **Limitación:** Asume relación lineal entre variables y precio.

---

## ⚙️ Pipeline de Procesamiento

1. **Carga de datos:** `casas.csv` con pandas.
2. **Separación de variables:** Features (X) = todas menos `precio`, Target (y) = `precio`.
3. **División:** Train/Test split con `train_test_split` y `random_state=42`.
4. **Entrenamiento:** `LinearRegression().fit(X_train, y_train)`
5. **Predicción:** `modelo.predict(X_test)`
6. **Evaluación:** MAE, RMSE, R².
7. **Serialización:** `joblib.dump(modelo, 'modelo_precios_casas.pkl')`
8. **Exportación:** Resultados comparativos en `resultados_predicciones.csv`.

---

## 📈 Métricas y Resultados

| Métrica | Descripción |
|---------|-------------|
| **MAE** | Error Absoluto Medio — promedio del error en unidades monetarias |
| **RMSE** | Raíz del Error Cuadrático Medio — penaliza errores grandes |
| **R²** | Coeficiente de determinación — qué proporción de la varianza explica el modelo |

> Con solo 5 muestras, las métricas son ilustrativas y no generalizables.

---

## 🔮 Cómo Usar el Modelo Entrenado

```python
import joblib

# Cargar el modelo
modelo = joblib.load("modelo_precios_casas.pkl")

# Nueva casa: [area_m2, habitaciones, baños, antigüedad, garajes]
nueva_casa = [[150, 3, 2, 5, 1]]

precio_predicho = modelo.predict(nueva_casa)
print(f"Precio estimado: ${precio_predicho[0]:,.0f}")
```

---

## ▶️ Ejecución

```bash
cd house_price_prediction
jupyter notebook house_price_prediction.ipynb
```

---

[⬅️ Volver al índice principal](../README.md)
