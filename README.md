# 🧠 Predicción de Fallos en Sensores (Mantenimiento Predictivo)

Proyecto de **Machine Learning** orientado a detectar y predecir fallos en equipos industriales a partir de lecturas de sensores.  
El objetivo es **anticipar incidencias y reducir paradas no planificadas**, optimizando el mantenimiento mediante un enfoque **predictivo**.


## 🎯 Objetivo y contexto

Este proyecto desarrolla un modelo capaz de **identificar con antelación cuándo un componente o sensor puede fallar**.  
En entornos industriales, detectar a tiempo un fallo potencial evita pérdidas de producción, reduce costes y permite planificar mantenimientos preventivos.

Dado que los **falsos negativos** (no detectar un fallo real) son los más costosos, se priorizan las métricas **Recall** y **F1-score**, equilibrando sensibilidad y precisión.


## 📊 Datos y variables

Dataset: [Kaggle – Machine Failure Prediction Using Sensor Data](https://www.kaggle.com/datasets/umerrtx/machine-failure-prediction-using-sensor-data/data)

**Variables:**

- Numéricas: `footfall`, `AQ`, `USS`, `CS`, `VOC`, `RP`, `IP`, `Temperature`
- Categórica: `tempMode`
- Objetivo: `fail` → 0 = no fallo, 1 = fallo

**Características:**

- Clases relativamente equilibradas (~58% / 42%)
- Sin valores nulos
- Outliers conservados al considerarse lecturas físicas plausibles


## 🔍 Análisis Exploratorio (EDA)

Hallazgos principales:

- **VOC**: mayor correlación con `fail` (~0.8).  
  - `VOC ≥ 6` aparece en la gran mayoría de los casos con fallo.
- **AQ**: segunda variable más influyente (r ≈ 0.58).  
  - `AQ > 5` se asocia con incremento claro de probabilidad de fallo.
- **footfall < 40** y **USS ≤ 2** muestran también mayor incidencia de fallos.
- No se detecta multicolinealidad extrema entre predictores.

> **Conclusión EDA:** VOC y AQ son sensores clave para anticipar fallos, apoyados por patrones en footfall y USS.

### Distribución de VOC según estado de fallo

![Boxplot VOC](docs/figures/Voc-Fail%20Boxplot.png)

### Frecuencia de VOC por clase

![Histograma VOC](docs/figures/VOC-Fail%20Histogramaa.png)

*(VOC elevado → mayor proporción de fallos detectados)*


## ⚙️ Preprocesamiento

El flujo de preprocesamiento se implementa mediante un **Pipeline de scikit-learn** para garantizar reproducibilidad y evitar fugas de información:

- **División estratificada 80/20** (train/test).
- **Eliminación de `tempMode`** por baja relevancia en este dataset.
- **Transformación logarítmica** en `footfall` (`log1p`) para reducir asimetría.
- **Ingeniería de características basada en el EDA**:
  - `flag_voc_ge6`
  - `flag_aq_gt5`
  - `flag_foot_lt40`
  - `flag_uss_le2`
- **Escalado robusto** con `RobustScaler` en variables numéricas.

Todo el preprocesado queda integrado en el Pipeline junto con el modelo final.


## 🤖 Modelado y evaluación

Modelos evaluados con **validación cruzada estratificada (K=5)**:

| Modelo              | F1 (media CV) |
|---------------------|---------------|
| Logistic Regression | 0.887         |
| SVM (RBF)           | **0.890**     |
| Random Forest       | 0.865         |
| Gradient Boosting   | 0.872         |
| Decision Tree       | 0.842         |

El modelo seleccionado es **SVM con kernel RBF**, por su mejor equilibrio entre Recall y Precision.


## 🧪 Resultados finales en test

Rendimiento del modelo SVM (RBF) optimizado en el conjunto de test:

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.9418 |
| Precision | 0.9250 |
| Recall    | **0.9367** |
| F1-score  | **0.9308** |
| ROC AUC   | 0.9770 |


El modelo detecta la mayoría de los fallos manteniendo un nivel bajo de falsas alarmas.  
Los resultados son coherentes con la validación cruzada, sin evidencias de sobreajuste.



## 💾 Guardado del modelo

El Pipeline completo (preprocesamiento + modelo SVM optimizado) se guarda como archivo `.pkl` para permitir su reutilización sin reentrenar:

```python
import joblib
joblib.dump(model_final, "models/sensorfail_svm_rbf_final.pkl")


### ▶️ Uso posterior del modelo guardado

Una vez entrenado y almacenado el pipeline, puede reutilizarse fácilmente en sesiones futuras sin necesidad de volver a entrenar:

```python
import joblib

### Cargar el modelo previamente guardado
pipeline = joblib.load("models/sensorfail_svm_rbf_final.pkl")

### Realizar predicciones sobre nuevos datos
predicciones = pipeline.predict(nuevos_datos)


## 📁 **Estructura del repositorio**

sensor-failure-ml-project/
├─ docs/
│  └─ figures/       # Visualizaciones y gráficos del proyecto
├─ models/           # Modelo final (.pkl)
├─ notebooks/        # Notebook principal (EDA + modelado)
├─ LICENSE
└─ README.md


## 🧩 **Conclusiones**

- VOC y AQ se consolidan como los sensores con mayor capacidad predictiva de fallo.
- El modelo SVM (RBF) alcanzó un F1 ≈ 0.93 y ROC AUC ≈ 0.98, mostrando un excelente equilibrio entre recall y precisión.
- La solución permite anticipar fallos con fiabilidad, contribuyendo a reducir paradas no planificadas y optimizando el mantenimiento predictivo.


## ✍️ **Autor**

Alejandro Álvarez Selva
Proyecto de Mantenimiento Predictivo mediante Machine Learning
LinkedIn: https://www.linkedin.com/in/alejandroaas1991
