# -*- coding: utf-8 -*-
"""
Perfiles de salud mediante clustering (K-means) — Aplicación Streamlit
Proyecto de aula · Minería de Datos (VIRTUAL-25786) · Universidad de la Costa (CUC)

La app aplica el modelo K-means (k = 2) entrenado en el corte 2 sobre el Dataset Sintético de Salud.
No usa scikit-learn en tiempo de ejecución: los parámetros del modelo (límites de Tukey, escalador,
centroides y PCA) se cargan desde model/model_params.json, lo que evita problemas de versiones.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="Perfiles de salud · K-means", page_icon="🩺", layout="wide")

# ----------------------------------------------------------------------------------------------
# Carga del modelo
# ----------------------------------------------------------------------------------------------
ETIQUETAS = {
    "Edad": "Edad (años)",
    "IMC": "IMC (kg/m²)",
    "Glucosa_mg_dL": "Glucosa (mg/dL)",
    "Presion_Sistolica_mmHg": "Presión sistólica (mmHg)",
    "Colesterol_mg_dL": "Colesterol (mg/dL)",
    "Horas_Sueno": "Horas de sueño",
    "Nivel_Estres": "Nivel de estrés (1–10)",
    "Consultas_Medicas": "Consultas médicas (12 meses)",
    "Actividad_Fisica_ord": "Actividad física (0–2)",
    "Fumador_bin": "Fumador (0/1)",
}
NOMBRES_GRUPO = {
    0: "Grupo 0 — mayor edad y valores clínicos más altos",
    1: "Grupo 1 — menor edad y valores clínicos más bajos",
}
COLORES = {"Grupo 0": "#c0504d", "Grupo 1": "#4f81bd"}
ACTIVIDAD = ["Baja", "Moderada", "Alta"]


@st.cache_resource
def cargar_modelo():
    with open(BASE / "model" / "model_params.json", encoding="utf-8") as f:
        params = json.load(f)
    ref = pd.read_csv(BASE / "model" / "reference_points.csv")
    return params, ref


P, REF = cargar_modelo()
VARS = P["vars"]
NUM = P["num_vars"]
MEDIA = np.array(P["scaler_mean"])
ESCALA = np.array(P["scaler_scale"])
CENT = np.array(P["centroides_z"])
PCA_MEAN = np.array(P["pca_mean"])
PCA_COMP = np.array(P["pca_components"])


def asignar(valores: dict):
    """Replica el pipeline del entrenamiento: winsorización -> estandarización -> centroide más cercano."""
    x = np.array([valores[v] for v in VARS], dtype=float)
    ajustadas = []
    for i, v in enumerate(VARS):
        if v in NUM:
            lo, hi = P["limites_tukey"][v]
            if x[i] < lo or x[i] > hi:
                ajustadas.append(v)
            x[i] = min(max(x[i], lo), hi)
    z = (x - MEDIA) / ESCALA
    dist = np.linalg.norm(CENT - z, axis=1)
    g = int(np.argmin(dist))
    otro = 1 - g
    margen = float((dist[otro] - dist[g]) / (dist[otro] + dist[g]))
    pca = (z - PCA_MEAN) @ PCA_COMP.T
    contrib = (z - CENT[otro]) ** 2 - (z - CENT[g]) ** 2  # >0: la variable acerca al grupo asignado
    return {"grupo": g, "z": z, "dist": dist, "margen": margen, "pca": pca,
            "contrib": contrib, "ajustadas": ajustadas}


# ----------------------------------------------------------------------------------------------
# Estado inicial de los controles y perfiles de ejemplo
# ----------------------------------------------------------------------------------------------
def _valores_de(perfil: dict) -> dict:
    return {
        "edad": int(round(perfil["Edad"])), "imc": round(float(perfil["IMC"]), 1),
        "glucosa": int(round(perfil["Glucosa_mg_dL"])), "presion": int(round(perfil["Presion_Sistolica_mmHg"])),
        "colesterol": int(round(perfil["Colesterol_mg_dL"])), "sueno": round(float(perfil["Horas_Sueno"]), 1),
        "estres": int(round(perfil["Nivel_Estres"])), "consultas": int(round(perfil["Consultas_Medicas"])),
        "actividad": ACTIVIDAD[int(round(perfil["Actividad_Fisica_ord"]))],
        "fumador": "Sí" if perfil["Fumador_bin"] >= 0.5 else "No",
    }


EJEMPLOS = {
    "Mediana del conjunto de datos": _valores_de(P["medianas"]),
    "Perfil promedio del Grupo 0": _valores_de(P["perfil"]["0"]),
    "Perfil promedio del Grupo 1": _valores_de(P["perfil"]["1"]),
}


def _cargar_ejemplo():
    for k, v in EJEMPLOS[st.session_state["ejemplo"]].items():
        st.session_state[k] = v


if "edad" not in st.session_state:
    for k, v in EJEMPLOS["Mediana del conjunto de datos"].items():
        st.session_state[k] = v

# ----------------------------------------------------------------------------------------------
# Barra lateral: propósito y advertencias
# ----------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("¿Para qué sirve esta app?")
    st.write(
        "Muestra **a cuál de dos perfiles de salud se parece más una persona**, según 10 indicadores "
        "clínicos y de hábitos. Es una herramienta **educativa** para entender cómo funciona una "
        "segmentación con K-means."
    )
    st.warning(
        "**No es una herramienta médica.** El modelo se entrenó con un dataset **sintético** (datos "
        "simulados con fines docentes). Los grupos describen datos, **no son diagnósticos** y no deben "
        "usarse para tomar decisiones clínicas."
    )
    st.caption(
        "Proyecto de aula · Minería de Datos (VIRTUAL-25786)  \nUniversidad de la Costa (CUC)  \n"
        "Profesor: José Escorcia-Gutierrez"
    )

# ----------------------------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------------------------
st.title("🩺 Perfiles de salud mediante clustering")
st.markdown(
    "Ingresa los datos de una persona (real o hipotética) y la aplicación calculará **a cuál de los dos "
    "grupos encontrados por K-means se parece más**, explicará **por qué** y mostrará **qué tan clara** "
    "es esa asignación."
)

tab_sim, tab_grupos, tab_modelo = st.tabs(["🔎 Simulador", "👥 Los dos grupos", "ℹ️ Sobre el modelo"])

# ==============================================================================================
# TAB 1 — SIMULADOR
# ==============================================================================================
with tab_sim:
    st.subheader("1. Ingresa los datos")
    st.selectbox(
        "Empieza desde un ejemplo (opcional)", list(EJEMPLOS), key="ejemplo", on_change=_cargar_ejemplo,
        help="Carga valores típicos para probar la app rápidamente. Luego puedes modificarlos.",
    )

    with st.form("formulario"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Indicadores clínicos**")
            edad = st.slider("Edad (años)", 18, 90, key="edad")
            imc = st.slider("Índice de masa corporal, IMC (kg/m²)", 15.0, 60.0, step=0.1, key="imc",
                            help="Peso en kg dividido por la estatura en metros al cuadrado.")
            glucosa = st.slider("Glucosa en ayunas (mg/dL)", 60, 340, key="glucosa")
            presion = st.slider("Presión arterial sistólica (mmHg)", 80, 230, key="presion",
                                help="El número más alto de la medición de presión arterial.")
            colesterol = st.slider("Colesterol total (mg/dL)", 100, 280, key="colesterol")
        with c2:
            st.markdown("**Hábitos y uso de servicios de salud**")
            sueno = st.slider("Horas de sueño por noche", 3.0, 11.0, step=0.1, key="sueno")
            estres = st.slider("Nivel de estrés percibido (1 = muy bajo, 10 = muy alto)", 1, 10, key="estres")
            consultas = st.slider("Consultas médicas en los últimos 12 meses", 0, 12, key="consultas")
            actividad = st.select_slider("Nivel de actividad física", options=ACTIVIDAD, key="actividad")
            fumador = st.radio("¿Es fumador?", ["No", "Sí"], horizontal=True, key="fumador")
        enviar = st.form_submit_button("Calcular grupo", type="primary", width="stretch")

    # Se calcula siempre (con los valores actuales) para que el resultado exista desde la primera carga
    valores = {
        "Edad": edad, "IMC": imc, "Glucosa_mg_dL": glucosa, "Presion_Sistolica_mmHg": presion,
        "Colesterol_mg_dL": colesterol, "Horas_Sueno": sueno, "Nivel_Estres": estres,
        "Consultas_Medicas": consultas, "Actividad_Fisica_ord": ACTIVIDAD.index(actividad),
        "Fumador_bin": 1 if fumador == "Sí" else 0,
    }
    r = asignar(valores)
    g = r["grupo"]

    st.subheader("2. Resultado")
    if r["margen"] < P["margen_p33"]:
        claridad = "Perfil intermedio: está cerca de la frontera entre los dos grupos"
    elif r["margen"] < P["margen_p66"]:
        claridad = "Asignación moderada"
    else:
        claridad = "Asignación clara"

    col_a, col_b = st.columns([3, 2])
    with col_a:
        (st.error if g == 0 else st.info)(f"### {NOMBRES_GRUPO[g]}")
        st.write(
            f"El perfil ingresado está **más cerca del centro del Grupo {g}** que del Grupo {1 - g} "
            f"(distancia estandarizada {r['dist'][g]:.2f} frente a {r['dist'][1 - g]:.2f})."
        )
    with col_b:
        st.metric("Claridad de la asignación", claridad)
        st.caption(
            "Los dos grupos se solapan bastante (ver «Sobre el modelo»), por lo que muchos perfiles quedan "
            "cerca de la frontera. Un perfil intermedio **no** indica nada clínico."
        )

    if r["ajustadas"]:
        st.caption(
            "ℹ️ Valores fuera del rango habitual del conjunto de datos, ajustados al límite durante el cálculo "
            "(igual que en el entrenamiento): " + ", ".join(ETIQUETAS[v] for v in r["ajustadas"]) + "."
        )

    # ---- Explicación: variables que más acercan al grupo asignado
    orden = np.argsort(-r["contrib"])
    top = [VARS[i] for i in orden[:3] if r["contrib"][i] > 0]
    if top:
        st.markdown("**¿Qué variables acercan más este perfil al grupo asignado?**")
        for v in top:
            i = VARS.index(v)
            media_g = P["perfil"][str(g)][v]
            valor = valores[v]
            if v == "Actividad_Fisica_ord":
                texto_v, texto_m = ACTIVIDAD[int(valor)], f"{media_g:.2f} en escala 0–2"
            elif v == "Fumador_bin":
                texto_v, texto_m = ("Sí" if valor else "No"), f"{media_g * 100:.0f} % de fumadores"
            else:
                texto_v, texto_m = f"{valor:g}", f"{media_g:.1f}"
            st.markdown(f"- **{ETIQUETAS[v]}**: tu valor ({texto_v}) se parece más al promedio del Grupo {g} ({texto_m}).")

    # ---- Gráfico 1: perfil estandarizado
    st.markdown("**Tu perfil frente a los dos grupos** (valores estandarizados: 0 = promedio de todos los registros)")
    filas = []
    for i, v in enumerate(VARS):
        filas.append({"variable": ETIQUETAS[v], "serie": "Tu perfil", "z": float(r["z"][i])})
        filas.append({"variable": ETIQUETAS[v], "serie": "Grupo 0", "z": float(CENT[0][i])})
        filas.append({"variable": ETIQUETAS[v], "serie": "Grupo 1", "z": float(CENT[1][i])})
    dfp = pd.DataFrame(filas)
    orden_vars = [ETIQUETAS[v] for v in VARS]
    colores = alt.Scale(domain=["Grupo 0", "Grupo 1", "Tu perfil"], range=["#c0504d", "#4f81bd", "#222222"])
    base = alt.Chart(dfp).encode(y=alt.Y("variable:N", sort=orden_vars, title=None))
    puntos = base.mark_point(filled=True, size=110, opacity=0.9).encode(
        x=alt.X("z:Q", title="Desviaciones estándar respecto al promedio general"),
        color=alt.Color("serie:N", scale=colores, title=None),
        shape=alt.Shape("serie:N", scale=alt.Scale(domain=["Grupo 0", "Grupo 1", "Tu perfil"],
                                                  range=["circle", "circle", "diamond"]), title=None),
        tooltip=["variable", "serie", alt.Tooltip("z:Q", format=".2f")],
    )
    cero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color="#999999", strokeDash=[4, 4]).encode(x="z:Q")
    st.altair_chart((cero + puntos).properties(height=330), width="stretch")
    st.caption(
        "Cada fila es una variable. Los círculos rojo y azul son el centro de cada grupo; el rombo negro es la "
        "persona ingresada. Cuanto más cerca está el rombo de un círculo, más se parece a ese grupo en esa variable."
    )

    # ---- Gráfico 2: mapa PCA
    st.markdown("**Dónde cae esta persona entre los 497 registros de entrenamiento** (proyección PCA)")
    ref = REF.copy()
    ref["Grupo"] = ref["grupo"].map({0: "Grupo 0", 1: "Grupo 1"})
    dispersion = alt.Chart(ref).mark_circle(size=35, opacity=0.45).encode(
        x=alt.X("pc1:Q", title=f"Componente 1 ({P['pca_var'][0] * 100:.1f} % de la varianza)"),
        y=alt.Y("pc2:Q", title=f"Componente 2 ({P['pca_var'][1] * 100:.1f} %)"),
        color=alt.Color("Grupo:N", scale=alt.Scale(domain=list(COLORES), range=list(COLORES.values()))),
        tooltip=["Grupo"],
    )
    yo = pd.DataFrame({"pc1": [float(r["pca"][0])], "pc2": [float(r["pca"][1])], "Punto": ["Tu perfil"]})
    marca = alt.Chart(yo).mark_point(shape="diamond", size=320, filled=True, color="#111111",
                                    stroke="white", strokeWidth=1.5).encode(x="pc1:Q", y="pc2:Q", tooltip=["Punto"])
    st.altair_chart((dispersion + marca).properties(height=380), width="stretch")
    st.caption(
        "Los dos grupos forman un gradiente continuo, no «islas» separadas: por eso la frontera es difusa. "
        f"Las dos componentes explican solo el {sum(P['pca_var']) * 100:.1f} % de la varianza total (proyección parcial)."
    )

    # ---- Tabla comparativa
    st.markdown("**Comparación en unidades originales**")
    filas_t = []
    for v in VARS:
        if v == "Actividad_Fisica_ord":
            val, m0, m1 = actividad, f"{P['perfil']['0'][v]:.2f}", f"{P['perfil']['1'][v]:.2f}"
        elif v == "Fumador_bin":
            val, m0, m1 = fumador, f"{P['perfil']['0'][v] * 100:.0f} %", f"{P['perfil']['1'][v] * 100:.0f} %"
        else:
            val, m0, m1 = f"{valores[v]:g}", f"{P['perfil']['0'][v]:.1f}", f"{P['perfil']['1'][v]:.1f}"
        filas_t.append({"Variable": ETIQUETAS[v], "Tu valor": val, "Promedio Grupo 0": m0, "Promedio Grupo 1": m1})
    st.dataframe(pd.DataFrame(filas_t), hide_index=True, width="stretch")

# ==============================================================================================
# TAB 2 — LOS DOS GRUPOS
# ==============================================================================================
with tab_grupos:
    st.subheader("Cómo son los dos grupos")
    st.write(
        f"K-means dividió los **{P['n_registros']} registros** en dos grupos: el **Grupo 0** con "
        f"**{P['tamanos']['0']}** personas ({P['tamanos']['0'] / P['n_registros'] * 100:.1f} %) y el **Grupo 1** con "
        f"**{P['tamanos']['1']}** ({P['tamanos']['1'] / P['n_registros'] * 100:.1f} %). Los números son identificadores "
        "arbitrarios del algoritmo; los nombres solo **describen los datos observados**."
    )
    filas_g = []
    for v in VARS:
        if v == "Fumador_bin":
            a, b = f"{P['perfil']['0'][v] * 100:.0f} %", f"{P['perfil']['1'][v] * 100:.0f} %"
        else:
            a, b = f"{P['perfil']['0'][v]:.2f}", f"{P['perfil']['1'][v]:.2f}"
        filas_g.append({"Variable": ETIQUETAS[v], "Grupo 0": a, "Grupo 1": b,
                        "Diferencia estandarizada (d)": f"{P['d_efecto'][v]:.2f}"})
    st.dataframe(pd.DataFrame(filas_g), hide_index=True, width="stretch")
    st.markdown(
        "- Los grupos se diferencian sobre todo en **edad, colesterol, IMC, presión sistólica, consultas médicas y glucosa**.\n"
        "- **Horas de sueño** y **nivel de estrés** casi no contribuyen a separarlos (|d| ≈ 0,10).\n"
        "- La *diferencia estandarizada* (d) mide cuánto se separan los promedios en desviaciones estándar; "
        "|d| ≥ 0,8 se considera grande.\n"
        "- Se usó la variable `Nivel_Riesgo` **solo como referencia externa** (nunca como entrada del modelo): "
        "el Grupo 0 concentra 59,3 % de riesgo Alto y el Grupo 1, 58,3 % de riesgo Bajo. Como esa variable "
        "se construyó con las mismas características, esto verifica la coherencia pero **no es una validación independiente**."
    )

# ==============================================================================================
# TAB 3 — SOBRE EL MODELO
# ==============================================================================================
with tab_modelo:
    st.subheader("Datos y método")
    st.markdown(
        f"- **Datos:** *Dataset Sintético de Salud* (500 registros, 14 variables, 2025), generado con fines docentes. "
        f"Tras eliminar 3 duplicados exactos quedaron **{P['n_registros']}** registros. No son pacientes reales.\n"
        "- **Variables del modelo (10):** edad, IMC, glucosa, presión sistólica, colesterol, horas de sueño, nivel de "
        "estrés, consultas médicas, actividad física y tabaquismo.\n"
        "- **Preparación:** faltantes imputados con la mediana/moda; valores extremos ajustados a los límites de Tukey "
        "(winsorización); estandarización (media 0, desviación 1), porque K-means usa distancia euclidiana.\n"
        f"- **Modelo:** K-means (k = {P['k']}, 50 inicializaciones, semilla {P['semilla']}). "
        "k = 2 se eligió por tener la mayor silueta y el mayor índice de Calinski-Harabasz, y por ser perfectamente "
        "reproducible entre semillas (ARI = 1,000)."
    )
    st.subheader("Métricas")
    m1, m2, m3 = st.columns(3)
    m1.metric("Coeficiente de silueta", f"{P['silueta']:.3f}")
    m2.metric("Inercia (WCSS)", f"{P['inercia']:.1f}")
    m3.metric("Estabilidad entre semillas (ARI)", "1,000")
    st.warning(
        f"**Lectura honesta de las métricas.** Una silueta de {P['silueta']:.3f} es baja: según Kaufman y Rousseeuw "
        "(1990), valores ≤ 0,25 indican que **no se encontró una estructura sustancial**. Es decir, los datos "
        "forman un continuo y K-means lo parte en dos, en lugar de descubrir grupos naturalmente separados. "
        "Por eso la app indica qué tan cerca de la frontera queda cada perfil. Como es aprendizaje no supervisado, "
        "no se reportan accuracy ni matriz de confusión."
    )
    st.subheader("Limitaciones")
    st.markdown(
        "- Datos **sintéticos** con asociaciones simuladas: los patrones no son evidencia sobre poblaciones reales.\n"
        "- Muestra pequeña (497 registros) y modelo con solo dos grupos.\n"
        "- K-means asume grupos de forma esférica y usa distancia euclidiana también con variables binarias y ordinales.\n"
        "- Los resultados son **asociaciones descriptivas**, no relaciones causales ni diagnósticos."
    )
    st.subheader("Equipo")
    st.write(
        "García López Jeffrey Daniel · Sánchez Cabrera Julián Andrés · Ballesteros Zabaleta Antonio José  \n"
        "Ingeniería de Sistemas, Universidad de la Costa (CUC) · Minería de Datos – Virtual · Profesor José Escorcia-Gutierrez"
    )
