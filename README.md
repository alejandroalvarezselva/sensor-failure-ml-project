# 🧠 Predicción de Fallos en Sensores (Mantenimiento Predictivo)

Proyecto de **Machine Learning** orientado a detectar y predecir fallos en equipos industriales a partir de lecturas de sensores.  
El objetivo es **anticipar incidencias y reducir paradas no planificadas**, optimizando el mantenimiento mediante un enfoque **predictivo**.


## 🎯 Objetivo y contexto

Este proyecto busca desarrollar un modelo capaz de **identificar con antelación cuándo un componente o sensor puede fallar**.  
En entornos industriales, **detectar a tiempo** un fallo potencial evita pérdidas, reduce costes y permite planificar mantenimientos preventivos.

Dado que los **falsos negativos** (no detectar un fallo real) son los más costosos, se priorizan las métricas **Recall** y **F1-score**, equilibrando sensibilidad y precisión.


## 📊 Datos y variables

El dataset proviene de [Kaggle – Machine Failure Prediction Using Sensor Data](https://www.kaggle.com/datasets/umerrtx/machine-failure-prediction-using-sensor-data/data).

**Principales variables:**
- Numéricas: `footfall`, `AQ`, `USS`, `CS`, `VOC`, `RP`, `IP`, `Temperature`
- Categórica: `tempMode`
- Variable objetivo: `fail` → (0 = no fallo, 1 = fallo)

**Características generales:**
- Balance de clases: ~58% sin fallo / ~42% con fallo  
- Sin valores nulos  
- Outliers plausibles (no eliminados)


## 🔍 Análisis Exploratorio (EDA)

Durante el análisis exploratorio se identificaron los siguientes patrones clave:

- **VOC**: correlación más fuerte con `fail` (~0.8)  
  - `VOC ≥ 6` está asociado a ~95% de los fallos.  
- **AQ**: segunda variable más influyente (r ≈ 0.58)  
  - `AQ > 5` incrementa notablemente la probabilidad de fallo.  
- **footfall < 40** y **USS ≤ 2** también mostraron tendencia a fallo.  

> En resumen, **VOC y AQ son las señales sensoriales más críticas** para anticipar un fallo.

### Distribución de VOC según estado de fallo
![Boxplot VOC](docs/figures/Voc-Fail%20Boxplot.png)

### Frecuencia de VOC por clase
![Histograma VOC](docs/figures/VOC-Fail%20Histogramaa.png)

*(VOC elevado → mayor proporción de fallos detectados)*


## ⚙️ Preprocesamiento

Para asegurar la consistencia entre entrenamiento y test se aplicó un flujo completo de preprocesamiento dentro de un **Pipeline de scikit-learn**:

- **División estratificada 80/20** para mantener equilibrio de clases.  
- **Selección de variables**: se eliminaron las no informativas (`tempMode`).  
- **Transformación logarítmica**: aplicada a `footfall` para estabilizar su escala.  
- **Creación de flags binarios**:
  - `flag_voc_ge6`
  - `flag_aq_gt5`
  - `flag_foot_lt40`
  - `flag_uss_le2`
- **Escalado robusto**: `RobustScaler` para reducir el impacto de outliers.  

El Pipeline evita fugas de información (*data leakage*) y permite reutilizar el modelo en producción sin repetir el preprocesamiento.


## 🤖 Modelado y evaluación

Se compararon los siguientes modelos mediante validación cruzada (K=5):

| Modelo | F1 (media CV) |
|:-------|:---------------|
| Logistic Regression | 0.887 |
| SVM (RBF) | **0.890** |
| Random Forest | 0.865 |
| Gradient Boosting | 0.872 |
| Decision Tree | 0.842 |

El modelo final elegido fue el **SVM con kernel RBF**, por ofrecer el mejor equilibrio entre precisión y recall.


## 🧪 Resultados finales en test

| Métrica | Valor |
|:--------|:------:|
| Accuracy | 0.9418 |
| Precision | 0.9250 |
| Recall | **0.9367** |
| F1-score | **0.9308** |
| ROC AUC | 0.9770 |


El modelo logra **detectar la mayoría de los fallos** (recall alto) manteniendo pocas falsas alarmas.  
Los resultados son consistentes con la validación cruzada, sin indicios de sobreajuste.


## 💾 Guardado del modelo

El pipeline completo (preprocesamiento + modelo SVM optimizado) se guardó como archivo `.pkl`, permitiendo su reutilización sin reentrenar:

```python
import joblib
joblib.dump(model_final, "models/sensorfail_svm_rbf_final.pkl")



## ✍️ Autor

**Alejandro Álvarez Selva**  
Proyecto de Mantenimiento Predictivo mediante Machine Learning
