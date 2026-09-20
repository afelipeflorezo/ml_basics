# 🎬 Sistema de Recomendación — Filtrado Colaborativo

Sistema de recomendación que sugiere productos a usuarios basándose en los patrones de calificaciones de usuarios similares, usando **similitud coseno** como medida de afinidad.

---

## 📁 Estructura del Módulo

```
sistema_recomendacion/
├── README.md                          ← Este archivo
├── sistema_recomendacion.ipynb        ← Notebook con el pipeline completo
├── calificaciones.csv                 ← Matriz de calificaciones usuario-producto
└── modelo_recomendador.pkl            ← Modelo con similitudes pre-calculadas
```

---

## 📊 Dataset: `calificaciones.csv`

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `usuario` | string | Nombre del usuario |
| `producto` | string | Nombre del producto |
| `calificacion` | float | Puntuación asignada (escala numérica) |

- **Total de registros:** 24 calificaciones
- **Formato:** Tabla larga (long format) que se pivotea a una **matriz usuario × producto**

---

## 🧮 Fundamento Teórico

### Filtrado Colaborativo basado en Ítems

El sistema recomienda productos similares a los que el usuario ya calificó positivamente.

1. **Matriz de calificaciones:** Se construye una tabla pivoteada donde filas = usuarios, columnas = productos, valores = calificaciones.
2. **Similitud Coseno:** Mide el ángulo entre dos vectores de producto:

```
                     A · B
cos(θ) = ──────────────────────
            ||A|| × ||B||
```

Un valor cercano a **1** indica productos con patrones de calificación similares.

3. **Recomendación:** Para un usuario dado, se buscan los productos más similares a los que ya le gustaron y que aún no ha calificado.

> **Ventaja:** No requiere información del contenido de los productos, solo comportamiento de usuarios.  
> **Desventaja:** Problema del arranque en frío (*cold start*) — no funciona bien con usuarios o productos nuevos sin calificaciones.

---

## ⚙️ Pipeline de Procesamiento

1. **Carga de datos:** `calificaciones.csv` con pandas.
2. **Pivoteo:** Tabla larga → Matriz usuario × producto (`pivot_table`).
3. **Similitud Coseno:** `cosine_similarity()` sobre la matriz transpuesta (producto × producto).
4. **Función de recomendación:** Dado un producto, devuelve los N más similares.
5. **Serialización:** `joblib.dump(modelo_recomendador, 'modelo_recomendador.pkl')`

---

## 📈 Resultados

El modelo genera una **matriz de similitud** entre todos los productos. Para cada producto, puede listar los más afines:

```
Productos similares a "Producto_A":
  1. Producto_C  (similitud: 0.92)
  2. Producto_E  (similitud: 0.87)
  3. Producto_B  (similitud: 0.75)
```

---

## 🔮 Cómo Usar el Modelo Entrenado

```python
import joblib

# Cargar modelo (incluye la matriz de similitudes y metadata)
modelo = joblib.load("modelo_recomendador.pkl")

# El modelo contiene:
# - Matriz de similitudes entre productos
# - Nombres de productos
# - Función o estructura para generar recomendaciones

# Ejemplo de uso (verificar estructura del pkl):
print(type(modelo))
print(modelo.keys() if isinstance(modelo, dict) else dir(modelo))
```

---

## ▶️ Ejecución

```bash
cd sistema_recomendacion
jupyter notebook sistema_recomendacion.ipynb
```

---

[⬅️ Volver al índice principal](../README.md)
