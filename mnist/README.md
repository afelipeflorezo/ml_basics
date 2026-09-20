# 🔢 MNIST — Clasificación de Dígitos con MLP y CNN (TensorFlow/Keras)

Proyecto de **Deep Learning** que compara dos arquitecturas de redes neuronales para clasificar dígitos manuscritos del dataset MNIST: una red **MLP (fully connected)** y una **CNN (convolucional)**.

---

## 📁 Estructura del Módulo

```
mnist/
├── README.md                 ← Este archivo
├── mnist_mlp.ipynb           ← Red MLP (Perceptrón Multicapa)
├── mnist_cnn.ipynb           ← Red CNN (Convolucional)
└── mnist_mlp_cnn.ipynb       ← Comparación directa MLP vs CNN
```

---

## 📊 Dataset: MNIST

El dataset **MNIST** (Modified National Institute of Standards and Technology) es el benchmark clásico de visión por computador:

| Característica | Valor |
|----------------|-------|
| **Imágenes** | Dígitos manuscritos (0-9) |
| **Resolución** | 28 × 28 píxeles, escala de grises |
| **Entrenamiento** | 60,000 imágenes |
| **Prueba** | 10,000 imágenes |
| **Clases** | 10 (dígitos 0 a 9) |

Se carga automáticamente desde `keras.datasets.mnist.load_data()`.

---

## 🧮 Fundamento Teórico

### MLP (Perceptrón Multicapa)

Red neuronal **densa** (fully connected) donde cada neurona se conecta con todas las de la capa siguiente:

```
Input (784) → Dense(128, ReLU) → Dense(10, Softmax) → Output
```

- **Preprocesamiento:** La imagen 28×28 se aplana a un vector de 784 píxeles.
- **Limitación:** No captura la estructura espacial 2D de la imagen.

### CNN (Red Neuronal Convolucional)

Red que aplica **filtros convolucionales** para detectar patrones locales (bordes, curvas, texturas):

```
Input (28×28×1) → Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Output
```

- **Ventaja:** Captura patrones espaciales jerárquicos (bordes → formas → dígitos).
- **Resultado:** Típicamente superior al MLP en tareas de visión.

---

## 📓 Los 3 Notebooks

### 1. `mnist_mlp.ipynb` — Solo MLP
- Construye y entrena un MLP con capas Dense.
- Normalización de píxeles a [0, 1].
- Evaluación con accuracy y loss en test.

### 2. `mnist_cnn.ipynb` — Solo CNN
- Construye y entrena una CNN con capas Conv2D + MaxPooling2D.
- Reshape de datos a (28, 28, 1) para canal de imagen.
- Evaluación con accuracy y loss en test.

### 3. `mnist_mlp_cnn.ipynb` — Comparación MLP vs CNN
- Entrena **ambos modelos** en el mismo notebook.
- Genera gráficos comparativos de accuracy y loss.
- Permite observar directamente la ventaja de CNN sobre MLP.

---

## ⚙️ Pipeline de Procesamiento (Común)

1. **Carga:** `keras.datasets.mnist.load_data()`.
2. **Normalización:** Píxeles de [0, 255] → [0, 1] (`/ 255.0`).
3. **Reshape:** MLP aplana a (784,); CNN mantiene (28, 28, 1).
4. **Compilación:** `optimizer='adam'`, `loss='sparse_categorical_crossentropy'`.
5. **Entrenamiento:** `model.fit()` con epochs configurables.
6. **Evaluación:** `model.evaluate()` sobre datos de prueba.

---

## 📈 Métricas y Resultados Esperados

| Modelo | Accuracy Aprox. | Descripción |
|--------|-----------------|-------------|
| **MLP** | ~97-98% | Buena base, pero limitada sin estructura espacial |
| **CNN** | ~99%+ | Superior gracias a filtros convolucionales |

> La CNN típicamente supera al MLP por 1-2 puntos porcentuales, demostrando la importancia de la arquitectura convolucional para datos de imagen.

---

## 🔮 Cómo Usar los Modelos

Los modelos en este módulo se entrenan y evalúan dentro del notebook usando TensorFlow/Keras. No se serializan con `joblib` como los demás proyectos, sino que se podrían guardar con:

```python
# Guardar modelo Keras
model.save("modelo_mnist_cnn.h5")

# Cargar modelo Keras
from tensorflow import keras
modelo = keras.models.load_model("modelo_mnist_cnn.h5")

# Predecir un nuevo dígito (28×28 píxeles)
import numpy as np
imagen = np.random.rand(1, 28, 28, 1)  # Ejemplo
prediccion = modelo.predict(imagen)
digito = np.argmax(prediccion)
print(f"Dígito predicho: {digito}")
```

---

## ▶️ Ejecución

```bash
cd mnist
jupyter notebook mnist_mlp_cnn.ipynb    # Comparación completa
# o individualmente:
jupyter notebook mnist_mlp.ipynb        # Solo MLP
jupyter notebook mnist_cnn.ipynb        # Solo CNN
```

> **Requisito:** TensorFlow 2.x instalado (`pip install tensorflow`).

---

[⬅️ Volver al índice principal](../README.md)
