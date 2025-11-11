# 🧠 Predicción de Fallos en Sensores (Mantenimiento Predictivo)

Proyecto de **Machine Learning** orientado a detectar fallos en equipos industriales mediante datos de sensores.  
El objetivo es anticipar incidencias y reducir paradas no planificadas aplicando **mantenimiento predictivo**.


## 🎯 Objetivo

Desarrollar un modelo capaz de identificar con antelación cuándo un sensor o componente puede fallar.  
Este tipo de predicción permite optimizar el mantenimiento, reducir costes y mejorar la fiabilidad de la operación.

📌 **Tipo de problema:** Clasificación binaria  
📌 **Variable objetivo:** `fail` (1 = fallo, 0 = no fallo)  
📌 **Métrica prioritaria:** *Recall* y *F1-score*


## 📊 Datos

Fuente: [Kaggle – Machine Failure Prediction using Sensor Data](https://www.kaggle.com/datasets/umerrtx/machine-failure-prediction-using-sensor-data/data)

**Variables principales:**
- Sensores numéricos: `footfall`, `AQ`, `USS`, `CS`, `VOC`, `RP`, `IP`, `Temperature`
- Categórica: `tempMode`
- Objetivo: `fail`


## 🔍 Análisis Exploratorio (EDA)

- **VOC** y **AQ** resultaron ser los sensores con mayor influencia en el fallo.  
- **VOC ≥ 6** y **AQ > 5** incrementan significativamente la probabilidad de fallo.  
- **footfall < 40** y **USS ≤ 2** también son indicadores relevantes.

📈 *Ejemplos de visualizaciones:*

![VOC vs Fail Boxplot](docs/figures/Voc-Fail%20Boxplot.png)
![VOC vs Fail Histograma](docs/figures/VOC-Fail%20Histogramaa.png)


## ⚙️ Modelado

Se probaron varios algoritmos con validación cruzada.  
El modelo **SVM (kernel RBF)** ofreció el mejor equilibrio entre *recall* y *precisión*.

| Modelo | F1 promedio |
|:--|:--:|
| Logistic Regression | 0.887 |
| Random Forest | 0.853 |
| Gradient Boosting | 0.862 |
| Decision Tree | 0.841 |
| **SVM (RBF)** | ⭐ **0.890** |


## 🧪 Resultados Finales (Test)

| Métrica | Valor |
|:--|:--:|
| Accuracy | 0.9418 |
| Precision | 0.9250 |
| Recall | **0.9367** |
| F1-score | **0.9308** |
| ROC AUC | 0.9770 |

📊 El modelo logra detectar la mayoría de los fallos (alto *recall*) manteniendo pocas falsas alarmas.  


## 💾 Guardado y uso del modelo

El pipeline completo (preprocesamiento + modelo SVM optimizado) se almacenó como archivo `.pkl` en la carpeta `models/`,  
permitiendo su reutilización sin necesidad de reentrenar.

```python
import joblib

# Guardado del modelo
joblib.dump(model_final, "models/sensorfail_svm_rbf_final.pkl")

# Carga y predicción con nuevos datos
pipeline = joblib.load("models/sensorfail_svm_rbf_final.pkl")
predicciones = pipeline.predict(nuevos_datos)
```
📁 Estructura del repositorio

sensor-failure-ml-project/
├─ docs/
│  └─ figures/         # Gráficos y visualizaciones
├─ models/             # Modelo final (.pkl)
├─ notebooks/          # Notebook principal (EDA + modelado)
├─ LICENSE
└─ README.md

🧩 Conclusiones
VOC y AQ son los sensores con mayor capacidad predictiva.

El modelo SVM (RBF) alcanzó un F1 ≈ 0.93 y ROC AUC ≈ 0.98, demostrando alta fiabilidad.

La solución permite anticipar fallos y reducir costes de mantenimiento no planificados.

✍️ Autor

Alejandro Álvarez Selva
📘 Proyecto de Mantenimiento Predictivo mediante Machine Learning
🔗 LinkedIn: www.linkedin.com/in/alejandroaas1991
