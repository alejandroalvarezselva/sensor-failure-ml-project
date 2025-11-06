# 🧠 Predicción de Fallos en Sensores (Mantenimiento Predictivo)

Proyecto de Machine Learning orientado a detectar y predecir fallos en equipos industriales a partir de lecturas de sensores.  
El objetivo es anticipar incidencias y reducir paradas no planificadas, mejorando la eficiencia mediante mantenimiento predictivo.

**Tipo de problema:** Clasificación binaria  
**Variable objetivo:** `fail` (1 = fallo, 0 = no fallo)

---

## 🧭 Objetivo y contexto

Este proyecto busca desarrollar un modelo capaz de identificar con antelación cuándo un sensor o componente puede fallar.  
En entornos industriales, detectar a tiempo un fallo potencial evita pérdidas, reduce costes de mantenimiento y permite programar intervenciones preventivas.  
Por ello, se priorizan métricas de **recall** (minimizar falsos negativos) y **F1-score**, equilibrando precisión y sensibilidad.

---

## 📊 Datos y variables

El dataset contiene lecturas de distintos sensores junto con una variable de salida (`fail`) que indica si hubo o no fallo.  
Las principales variables son:

- **Sensores numéricos:** `footfall`, `AQ`, `USS`, `CS`, `VOC`, `RP`, `IP`, `Temperature`  
- **Categórica:** `tempMode`  
- **Variable objetivo:** `fail` (0 = no fallo, 1 = fallo)

Los datos se cargan desde Google Drive en el notebook de Colab.  
Cada usuario debe ajustar su ruta de acceso si desea replicar el proyecto.

---

## 🔍 Análisis Exploratorio (EDA)

- **Balance de clases:** relativamente equilibrado (~58% sin fallo / 42% con fallo), por lo que no se aplicaron técnicas de re-muestreo.  
- **Valores nulos:** inexistentes; **duplicados:** se eliminó 1 registro.  
- **Outliers:** detectados principalmente en `footfall` (16%), `CS` (9%) y `Temperature` (5%). Se conservaron al considerarse valores plausibles del proceso físico.  
- **Correlaciones:** `VOC` mostró la relación más fuerte con `fail` (~0.8), seguida de `AQ` (~0.58) y `Temperature` (~0.19). No se observaron correlaciones entre predictores superiores a 0.9.  
- **Umbrales críticos identificados:**  
  - `VOC ≥ 6`: asociado a la mayoría de los fallos (~95%)  
  - `AQ > 5`: incrementa el riesgo de fallo  
  - `footfall < 40`: tendencia a fallo  
  - `USS` en valores 1–2: indicativo de anomalía  

**Conclusión:** `VOC` y `AQ` son las variables más influyentes.  
`footfall` mostró una distribución muy asimétrica, con valores extremos y dispersión significativa, por lo que se aplicó una transformación logarítmica (`log1p`) para estabilizar su escala y mejorar la capacidad predictiva del modelo.

---

## ⚙️ Preprocesamiento

Se definió un flujo de transformación robusto para garantizar coherencia entre entrenamiento y test:

1. **División estratificada 80/20:** se asegura que las proporciones de las clases `fail=0` y `fail=1` sean similares en ambos conjuntos, evitando sesgos en la evaluación.  
2. **Selección de variables:** se eliminaron las no informativas, como `tempMode`, que no presentaba relación con el estado de fallo.  
3. **Ingeniería de características:**  
   - Transformación logarítmica en `footfall`.  
   - Creación de banderas binarias (`flag_voc_ge6`, `flag_aq_gt5`, `flag_foot_lt40`, `flag_uss_le2`) para capturar los patrones detectados en el EDA.  
4. **Escalado y codificación:**  
   - `RobustScaler` en variables numéricas para mitigar el efecto de outliers.  
   - `OneHotEncoder` preparado para categóricas si se incorporan en futuras versiones.  
5. **Pipeline completo:** se integró todo el preprocesamiento dentro de un `Pipeline` de scikit-learn, evitando fugas de información (data leakage).

---

## 🤖 Modelado y comparación

Se evaluaron cinco algoritmos de clasificación con validación cruzada estratificada (K=5):  
Regresión Logística, SVM, Random Forest, Gradient Boosting y Decision Tree.  
Las métricas consideradas fueron: accuracy, precision, recall, F1 y ROC AUC.

**Resultados promedio (CV):**
- **Logistic Regression (scaled):** F1 ≈ 0.887  
- **SVM (RBF, scaled):** F1 ≈ 0.884  
- Los modelos basados en árboles mostraron menor equilibrio entre recall y precisión.

Se seleccionaron **Regresión Logística** y **SVM** como modelos candidatos para optimización.

---

## 🎯 Optimización de hiperparámetros

Se aplicó `GridSearchCV` optimizando la métrica **F1**.  
Los mejores resultados fueron:

- **SVM (RBF):** `C=1`, `gamma='auto'`, F1 ≈ 0.890  
- **Logistic Regression:** `C=1`, `penalty='l2'`, `solver='lbfgs'`, F1 ≈ 0.887  

El modelo final elegido fue **SVM (RBF)** por ofrecer el mejor equilibrio entre recall y precisión.

---

## 🧪 Evaluación final en test

**SVM (RBF) optimizado – conjunto de test:**
- Accuracy: 0.9418  
- Precision: 0.9250  
- Recall: 0.9367  
- F1-score: 0.9308  
- ROC AUC: 0.9770  

El modelo logra detectar la mayoría de los fallos (recall alto) manteniendo pocas falsas alarmas.  
Los resultados son consistentes con la validación cruzada, sin evidencia de sobreajuste.

---

## 💾 Guardado del modelo

El pipeline final (preprocesamiento + modelo SVM) se guardó como archivo `.pkl` para permitir su reutilización sin reentrenar.  
Cada usuario puede ajustar la ruta de guardado según su entorno.

---

## 📁 Estructura del repositorio

sensor-failure-ml-project/
├─ data/ # Documentación de los datos y muestra
├─ docs/
│ └─ figures/ # Imágenes y visualizaciones
├─ models/ # Modelos entrenados (.pkl)
├─ notebooks/ # Notebook principal del proyecto
├─ src/ # Código modular (funciones y scripts)
├─ .gitignore
├─ LICENSE
└─ README.md

---

## ✍️ Autor

**Alejandro Álvarez Selva**  
Proyecto de Mantenimiento Predictivo mediante Machine Learning
