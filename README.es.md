# EDA de Precios de Airbnb en Nueva York

> Análisis exploratorio de datos sobre los listados de Airbnb en la Ciudad de Nueva York — identificando las variables que más influyen en el precio de los alojamientos en los cinco condados.

---

## Problema

¿Qué factores determinan el precio de un alojamiento en Airbnb en la Ciudad de Nueva York? Este proyecto realiza un EDA estructurado de extremo a extremo sobre los datos abiertos de Airbnb NYC para responder esa pregunta — limpiando los datos, explorando distribuciones y relaciones, gestionando valores atípicos y entregando un dataset con selección de características listo para modelado posterior.

## Conjunto de datos

- **Fuente:** [NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) — Kaggle
- **Características clave analizadas:** `neighbourhood_group`, `neighbourhood`, `room_type`, `price`, `minimum_nights`, `number_of_reviews`, `calculated_host_listings_count`, `availability_365`
- **Eliminadas en limpieza:** `id`, `name`, `host_name`, `last_review`, `reviews_per_month` — no relevantes para la predicción de precios
- **Variable objetivo:** `price`

## Metodología

**Paso 1 — Limpieza de datos**
Se verificaron cero IDs duplicados y cero valores nulos en todas las columnas retenidas. Se eliminaron cinco columnas de identificadores y metadatos irrelevantes.

**Paso 2 — Análisis univariante**
Se graficaron histogramas y diagramas de caja para todas las variables numéricas (`price`, `minimum_nights`, `number_of_reviews`, `calculated_host_listings_count`) y distribuciones de conteo para variables categóricas (`neighbourhood_group`, `room_type`).

**Paso 3 — Análisis multivariante**
Se construyeron gráficos de regresión y mapas de calor de correlación por pares para variables numéricas frente al precio. Se graficó el conteo de tipos de habitación desglosado por condado. Se factorizaron las columnas categóricas y se generó un mapa de calor de correlación completo y un pair plot.

**Paso 4 — Gestión de valores atípicos**
Se identificaron tres variables con mayores valores atípicos mediante inspección de boxplots:
- `price`: eliminados listados con precio $0
- `minimum_nights`: limitado a 15 noches (límite superior basado en IQR)
- `number_of_reviews`: documentado mediante análisis IQR

**Paso 5 — Escalado de características**
Se aplicó `MinMaxScaler` a todas las variables numéricas: `number_of_reviews`, `minimum_nights`, `calculated_host_listings_count`, `availability_365`, `neighbourhood_group`, `room_type`.

**Paso 6 — Selección de características**
Se utilizó `SelectKBest` con puntuación chi-cuadrado para identificar las 4 características más predictivas del precio en el conjunto escalado. Se guardaron las divisiones finales entrenamiento/prueba como `data/processed/clean_train_data.csv` y `clean_test_data.csv`, listas para modelado.

## Resultados

El EDA identificó `neighbourhood_group` y `room_type` como los principales impulsores categóricos del precio — consistente con la intuición del mundo real (Manhattan vs. el Bronx; casa completa vs. habitación compartida). El dataset de salida limpio y con selección de características proporciona una base sólida para un modelo de regresión que prediga el precio de un listado de Airbnb.

## Tecnologías utilizadas

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn`

## Ejecución local

```bash
git clone https://github.com/matthewkane-ml/EDA_Project.git
cd EDA_Project
pip install -r requirements.txt

# Añade el dataset crudo a data/raw/raw_data.csv, luego ejecuta:
python src/app.py
```

## Próximos pasos

- Construir un modelo de regresión (Regresión Lineal o Random Forest Regressor) sobre los datos limpios para predecir efectivamente el precio del alojamiento
- Investigar más a fondo los patrones geográficos de precios usando datos de latitud/longitud y una visualización de mapa (ej. Folium o Plotly)
- Realizar un análisis más profundo de los valores atípicos en `price` — muchos alojamientos en NYC son legítimamente caros, por lo que un límite IQR estricto podría eliminar puntos de datos válidos

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-kane-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
