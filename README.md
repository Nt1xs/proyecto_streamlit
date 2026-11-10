# 🩺 Perfiles de salud mediante clustering (K-means) — App en Streamlit

Despliegue del modelo entrenado en el **segundo corte** del proyecto de aula de **Minería de Datos**
(VIRTUAL-25786), Universidad de la Costa (CUC).

**Integrantes:** García López Jeffrey Daniel · Sánchez Cabrera Julián Andrés · Ballesteros Zabaleta Antonio José  
**Profesor:** José Escorcia-Gutierrez · **Programa:** Ingeniería de Sistemas

🔗 **App pública:** [](https://perfiles-salud-kmeans.streamlit.app/)

## ¿Qué hace la app?

Ingresas 10 indicadores de una persona (edad, IMC, glucosa, presión sistólica, colesterol, horas de sueño,
estrés, consultas médicas, actividad física y tabaquismo) y la app:

1. **Asigna** el perfil a uno de los dos grupos encontrados por K-means.
2. **Explica por qué**: variables que más lo acercan al grupo y comparación con el centro de cada grupo.
3. **Indica qué tan clara es la asignación** (muchos perfiles quedan cerca de la frontera).
4. **Ubica** a la persona en un mapa PCA con los 497 registros de entrenamiento.

> ⚠️ Herramienta **educativa**. Se entrenó con un dataset **sintético**; los grupos describen datos y **no son diagnósticos**.

## El modelo

| Elemento | Detalle |
|---|---|
| Datos | Dataset Sintético de Salud (500 registros → 497 tras eliminar 3 duplicados exactos) |
| Variables | 10 (8 numéricas + actividad física ordinal 0–2 + fumador 0/1) |
| Preparación | Imputación (mediana/moda) → winsorización a límites de Tukey → `StandardScaler` |
| Algoritmo | K-means, k = 2, `n_init = 50`, `random_state = 42` |
| Métricas | Silueta 0,151 · Inercia 4145,9 · ARI entre semillas 1,000 |

El coeficiente de silueta es bajo (≤ 0,25 = sin estructura sustancial según Kaufman y Rousseeuw, 1990): los datos
forman un continuo que K-means divide en dos. La app lo comunica de forma explícita. Por ser aprendizaje no
supervisado no se reportan accuracy ni matriz de confusión.

## Estructura del repositorio

```
app.py                      # Interfaz Streamlit
train_model.py              # Reproduce el modelo del notebook y exporta los parámetros
model/model_params.json     # Parámetros del modelo (límites, escalador, centroides, PCA, perfiles)
model/reference_points.csv  # Coordenadas PCA y grupo de los 497 registros (sin identificadores)
requirements.txt
.streamlit/config.toml
```

La app **no depende de scikit-learn** al ejecutarse: reimplementa la predicción (winsorizar → estandarizar →
centroide más cercano) con los parámetros exportados. Se verificó que coincide en el **100 %** de los 497 registros
con `KMeans.predict` de scikit-learn.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para regenerar el modelo (necesita el archivo `Dataset_Sintetico_Salud.xlsx` y `scikit-learn`, `openpyxl`):

```bash
pip install scikit-learn openpyxl
python train_model.py Dataset_Sintetico_Salud.xlsx
```

## Desplegar en Streamlit Community Cloud

1. Subir esta carpeta a un repositorio **público** de GitHub (rama `main`).
2. Entrar a <https://share.streamlit.io> con la cuenta de GitHub → **Create app**.
3. Elegir el repositorio, rama `main` y archivo principal `app.py` → **Deploy**.
4. Copiar el enlace `https://<nombre>.streamlit.app` y pegarlo arriba y en Moodle.

## Uso de inteligencia artificial

Se usó Claude (Anthropic) para apoyar la construcción de la app y de este repositorio. El detalle (prompts,
problemas identificados, ajustes y decisión final) se declara en el informe del proyecto.
