# -*- coding: utf-8 -*-
"""
train_model.py — Reproduce el modelo K-means del notebook del corte 2 y exporta sus parámetros.

Pipeline (idéntico al notebook Avance_Corte2_Mineria_Datos.ipynb):
  1. Eliminar duplicados exactos (500 -> 497 registros).
  2. Codificar Actividad_Fisica (Baja=0, Moderada=1, Alta=2) y Fumador (Sí=1, No=0).
  3. Imputar faltantes: mediana (numéricas) / moda (ordinal y binaria).
  4. Winsorizar las 8 variables numéricas a los límites de Tukey (Q1-1.5*IQR, Q3+1.5*IQR).
  5. Estandarizar con StandardScaler.
  6. K-means con k = 2, n_init = 50, random_state = 42.

Salidas (carpeta model/):
  - model_params.json     : todo lo necesario para predecir sin depender de scikit-learn.
  - reference_points.csv  : coordenadas PCA y clúster de los 497 registros (sin identificadores).

Uso:  python train_model.py [ruta_al_xlsx]      (por defecto: Dataset_Sintetico_Salud.xlsx)
Requiere: pandas, numpy, scikit-learn, openpyxl.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
K = 2
ARCHIVO = sys.argv[1] if len(sys.argv) > 1 else "Dataset_Sintetico_Salud.xlsx"

NUM = ["Edad", "IMC", "Glucosa_mg_dL", "Presion_Sistolica_mmHg", "Colesterol_mg_dL",
       "Horas_Sueno", "Nivel_Estres", "Consultas_Medicas"]
VARS = NUM + ["Actividad_Fisica_ord", "Fumador_bin"]

# 1) Datos y limpieza -------------------------------------------------------------------------
df = pd.read_excel(ARCHIVO, sheet_name="Datos")
df = df.drop_duplicates().reset_index(drop=True)
df["Actividad_Fisica_ord"] = df["Actividad_Fisica"].map({"Baja": 0, "Moderada": 1, "Alta": 2})
df["Fumador_bin"] = (df["Fumador"] == "Sí").astype(int)
X = df[VARS].copy()

# 2) Imputación -------------------------------------------------------------------------------
imput = {}
for c in VARS:
    imput[c] = float(X[c].median() if c in NUM else X[c].mode()[0])
    X[c] = X[c].fillna(imput[c])

# 3) Winsorización (límites de Tukey) ----------------------------------------------------------
limites = {}
Xw = X.copy()
for c in NUM:
    q1, q3 = X[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    limites[c] = [lo, hi]
    Xw[c] = X[c].clip(lo, hi)

# 4) Escalamiento y K-means ---------------------------------------------------------------------
scaler = StandardScaler()
Z = scaler.fit_transform(Xw)
km = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(Z)
labels = km.labels_
sil = float(silhouette_score(Z, labels))
pca = PCA(n_components=2, random_state=SEED).fit(Z)
P = pca.transform(Z)

# 5) Margen de asignación (qué tan cerca de la frontera está cada registro) ---------------------
D = np.linalg.norm(Z[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
d_prop = D[np.arange(len(Z)), labels]
d_otro = D[np.arange(len(Z)), 1 - labels]
margen = (d_otro - d_prop) / (d_otro + d_prop)

# 6) Perfil de cada clúster (unidades originales, matriz imputada sin winsorizar, como el notebook)
perfil = {int(c): X[labels == c].mean().round(4).to_dict() for c in range(K)}
d_efecto = ((X[labels == 1].mean() - X[labels == 0].mean()) / X.std()).round(3).to_dict()

params = {
    "vars": VARS,
    "num_vars": NUM,
    "imputacion": imput,
    "limites_tukey": limites,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "centroides_z": km.cluster_centers_.tolist(),
    "pca_mean": pca.mean_.tolist(),
    "pca_components": pca.components_.tolist(),
    "pca_var": pca.explained_variance_ratio_.tolist(),
    "rangos_datos": {c: [float(X[c].min()), float(X[c].max())] for c in VARS},
    "medianas": {c: float(X[c].median()) for c in VARS},
    "tamanos": {int(c): int((labels == c).sum()) for c in range(K)},
    "perfil": perfil,
    "d_efecto": d_efecto,
    "inercia": float(km.inertia_),
    "silueta": sil,
    "n_registros": int(len(df)),
    "margen_p33": float(np.quantile(margen, 0.33)),
    "margen_p66": float(np.quantile(margen, 0.66)),
    "semilla": SEED,
    "k": K,
}

os.makedirs("model", exist_ok=True)
with open("model/model_params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False, indent=2)
pd.DataFrame({"pc1": P[:, 0].round(4), "pc2": P[:, 1].round(4), "grupo": labels}).to_csv(
    "model/reference_points.csv", index=False)

# 7) Verificación contra las cifras del notebook / informe ---------------------------------------
print(f"Registros: {len(df)} | inercia: {km.inertia_:.1f} | silueta: {sil:.3f}")
print("Tamaños:", params["tamanos"], "| PCA var:", [round(v, 3) for v in params["pca_var"]])
print("Edad media por clúster:", {c: round(perfil[c]["Edad"], 2) for c in perfil})
print("Parámetros exportados en model/model_params.json")
