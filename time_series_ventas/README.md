# 📈 Pronóstico de Series Temporales de Ventas

Modelo de regresión que predice valores futuros de una serie temporal de ventas diarias, usando **features de rezago (lags)** con Regresión Lineal.

---

## 📁 Estructura del Módulo

```
time_series_ventas/
├── README.md                              ← Este archivo
├── time_series_ventas.ipynb               ← Notebook con el pipeline completo
├── serie_temporal.csv                     ← Serie temporal original (180 días)
├── resultados_series_temporales.csv       ← Comparación real vs predicho
├── predicciones_futuras.csv               ← Pronósticos a futuro
└── modelo_series_temporales.pkl           ← Modelo de regresión entrenado
```

---

## 📊 Dataset: `serie_temporal.csv`

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `fecha` | date | Fecha del registro |
| `valor` | float | Valor de ventas del día |

- **Total de registros:** 180 días de datos de ventas
- **Granularidad:** Diaria

---

## 🧮 Fundamento Teórico

### Regresión con Features de Rezago (Lag Features)

En lugar de usar un modelo de series temporales clásico (ARIMA, Prophet), este enfoque transforma el problema temporal en un **problema de regresión tabular**:

1. Se crean columnas de **rezago** (lag): el valor de ventas de hace 1, 2, ..., N días.
2. Estas columnas se usan como variables predictoras (features).
3. Se aplica **Regresión Lineal** estándar sobre las features de lag.

```
Ejemplo con 3 lags:

| fecha      | valor | lag_1 | lag_2 | lag_3 |
|------------|-------|-------|-------|-------|
| 2024-01-04 | 150   | 120   | 130   | 100   |
| 2024-01-05 | 160   | 150   | 120   | 130   |
```

> **Ventaja:** Simple, interpretable, usa herramientas de ML estándar.  
> **Limitación:** No captura estacionalidad compleja ni tendencias no lineales.

---

## ⚙️ Pipeline de Procesamiento

1. **Carga de datos:** `serie_temporal.csv` con pandas.
2. **Ingeniería de features:** Creación de columnas de rezago (`shift`).
3. **Limpieza:** Eliminación de filas con NaN producidos por los lags.
4. **División temporal:** Train/Test split respetando el orden cronológico.
5. **Entrenamiento:** `LinearRegression().fit(X_train, y_train)`
6. **Evaluación:** MAE, RMSE, R².
7. **Predicción futura:** Se usa el modelo iterativamente para generar pronósticos.
8. **Serialización:** `joblib.dump(modelo, 'modelo_series_temporales.pkl')`
9. **Exportación:** `predicciones_futuras.csv` y `resultados_series_temporales.csv`.

---

## 📈 Métricas y Resultados

| Métrica | Descripción |
|---------|-------------|
| **MAE** | Error Absoluto Medio — error promedio en unidades de venta |
| **RMSE** | Raíz del Error Cuadrático Medio — penaliza errores grandes |
| **R²** | Coeficiente de determinación — proporción de varianza explicada |

### Archivos de Resultados

- `resultados_series_temporales.csv`: 35 registros con `fecha`, `valor_real`, `valor_predicho`
- `predicciones_futuras.csv`: 3 predicciones a futuro con `fecha` y `valor_predicho`

---

## 🔮 Cómo Usar el Modelo Entrenado

```python
import joblib
import numpy as np

# Cargar el modelo
modelo = joblib.load("modelo_series_temporales.pkl")

# Para predecir el próximo día, necesitas los últimos N valores (lags)
# Ejemplo con 3 lags: [valor_ayer, valor_anteayer, valor_hace_3_dias]
ultimos_valores = np.array([[155.0, 148.0, 142.0]])

prediccion = modelo.predict(ultimos_valores)
print(f"Ventas predichas para mañana: {prediccion[0]:.2f}")
```

---

## ▶️ Ejecución

```bash
cd time_series_ventas
jupyter notebook time_series_ventas.ipynb
```

---

[⬅️ Volver al índice principal](../README.md)
