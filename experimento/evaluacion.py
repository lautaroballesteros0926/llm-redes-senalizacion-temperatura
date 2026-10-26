"""
Funciones de evaluación para nuestro experimento.

Qué es este archivo
-------------------
El repo del paper (saucermanlab/LLM-network-generation, licencia MIT) calcula recall y
precisión DENTRO del cuaderno `figure_generation.ipynb`, así que no se pueden importar.
Aquí copiamos esa lógica (celdas 36-38 del cuaderno) sin cambiar el criterio, solo
separada en funciones para poder usarla con una réplica a la vez.

Lo que SÍ se importa del repo (sin copiarlo): get_model_data, extract_AND_reactions y
compare_connections. El repo no se modifica: se desactiva la escritura de __pycache__ y
solo se cambia de carpeta de trabajo de forma temporal porque el repo usa rutas relativas
("models/...", "outputs/...").

Única diferencia con el original: si una división es 0/0 (p. ej. una réplica sin
predicciones) devolvemos NaN en vez de dejar que numpy dé un aviso.
"""
import sys
sys.dont_write_bytecode = True   # evita crear carpetas __pycache__ dentro del repo

import os, json, contextlib, io
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
REPO = RAIZ / "LLM-network-generation"
sys.path.insert(0, str(REPO))

from network_extraction import get_model_data, extract_AND_reactions      # noqa: E402  (del repo)
from predicted_connection_analysis import compare_connections as _compare_connections  # noqa: E402


@contextlib.contextmanager
def en_repo(silencio=True):
    """Cambia temporalmente el directorio de trabajo al del repo (rutas relativas)."""
    anterior = os.getcwd()
    os.chdir(REPO)
    try:
        if silencio:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        else:
            yield
    finally:
        os.chdir(anterior)


def tabla_aciertos(red, llm):
    """compare_connections del repo con los JSON del manuscrito (10 réplicas)."""
    with en_repo():
        return _compare_connections(None, red, True, llm)


def tabla_aciertos_archivo(ruta_json, red):
    """compare_connections del repo para UN archivo JSON nuestro."""
    with en_repo():
        return _compare_connections(str(ruta_json), red, False, None)


def conteo_aciertos(tabla):
    """Réplicas -> número de flechas verdaderas halladas (igual que performance_table del cuaderno)."""
    t = tabla.drop_duplicates(subset=["source", "target", "action"])
    cols = [c for c in t.columns if c in [str(i) for i in range(1, 11)]]
    return {c: int(t[c].sum()) for c in cols}, len(t)


# ---------- matrices nodo x nodo (cuaderno, celdas 36-38) ----------

def matriz_verdad(red):
    """Matriz de la red hecha a mano: 1 estimula, -1 inhibe, 0 sin flecha. (copia de la celda 36)"""
    with en_repo():
        df_species, df_reactions = get_model_data(red)
    vacia = pd.DataFrame(0, index=df_species["ID"].unique(), columns=df_species["ID"].unique())
    and_rx, non_and_rx = extract_AND_reactions(df_reactions)
    non_and_rx = [r for r in non_and_rx if r[0] != ""]   # quita las entradas (inputs) sin origen
    gt = vacia.copy()
    for _, rx in and_rx.items():
        for x in range(len(rx["sources"])):
            gt.loc[rx["sources"][x], rx["target"]] = 1 if rx["action"][x] == "stimulation" else -1
    for source, target, action in non_and_rx:
        gt.loc[source, target] = 1 if action == "stimulation" else -1
    return gt, vacia


def _mapa_genes(red):
    with en_repo():
        df_species, _ = get_model_data(red)
    mapa, nodos = {}, set()
    for _, row in df_species.iterrows():
        if type(row["gene_name"]) is not float:
            for gene in row["gene_name"].split(";"):
                mapa[gene.strip().lower()] = row["ID"]
                nodos.add(row["ID"])
    return mapa, nodos


def matriz_prediccion(interacciones, red, vacia):
    """
    interacciones: DataFrame con source_node, target_node, action (genes, tal como los da el LLM).
    Devuelve la matriz de predicción (copia de la celda 37, para una sola réplica).
    """
    mapa, nodos = _mapa_genes(red)
    df = pd.DataFrame(interacciones).copy()
    df["source_node"] = df["source_node"].astype(str).str.strip().str.lower().map(mapa)
    df["target_node"] = df["target_node"].astype(str).str.strip().str.lower().map(mapa)

    # caso especial del repo: en fibroblast/mechanosignaling AGT y AngII comparten gene_name 'agt'
    if red in ("fibroblast", "mechanosignaling") and mapa.get("agt") == "AGT":
        extra = []
        for _, row in df.iterrows():
            if row.get("source_node") == "AGT":
                nr = row.copy(); nr["source_node"] = "AngII"; extra.append(nr.to_dict())
            if row.get("target_node") == "AGT":
                nr = row.copy(); nr["target_node"] = "AngII"; extra.append(nr.to_dict())
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    pm = vacia.copy()
    for _, row in df.iterrows():
        s, t, a = row["source_node"], row["target_node"], row["action"]
        if s in nodos and t in nodos:   # solo flechas entre nodos que existen en el modelo
            pm.loc[s, t] = 1 if a == "stimulation" else -1 if a == "inhibition" else 0
    return pm


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def metricas(pm, gt):
    """Recall (sensibilidad), precisión, etc. de UNA réplica. (copia de la celda 38)"""
    tp = (((pm == 1) & (gt == 1)) | ((pm == -1) & (gt == -1)))
    tn = ((pm == 0) & (gt == 0))
    fp = ((((pm == 1) | (pm == -1)) & (gt == 0)) | ((pm == 1) & (gt == -1)) | ((pm == -1) & (gt == 1)))
    fn = ((pm == 0) & ((gt == 1) | (gt == -1)))
    TP, TN, FP, FN = (int(np.sum(x.values)) for x in (tp, tn, fp, fn))
    recall = _div(TP, TP + FN)
    precision = _div(TP, TP + FP)
    # extra nuestro (no está en el repo): flechas verdaderas que el LLM puso con el signo contrario
    signo_contrario = int(np.sum((((pm == 1) & (gt == -1)) | ((pm == -1) & (gt == 1))).values))
    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "recall": recall, "precision": precision,
        "especificidad": _div(TN, TN + FP),
        "F1": _div(2 * precision * recall, precision + recall) if TP else float("nan"),
        "verdad_total": int(np.sum((gt != 0).values)),
        "signo_contrario": signo_contrario,
    }


def cargar_jsons_manuscrito(red, llm):
    """{indice_replica(0..9): DataFrame} con la misma regla de nombres que el repo."""
    carpeta = REPO / "outputs" / "saved_jsons" / red / llm
    out = {}
    for archivo in sorted(carpeta.iterdir()):
        if archivo.is_file() and ".DS_Store" not in archivo.name:
            with open(archivo, encoding="utf-8") as f:
                out[int(archivo.name.split("_")[3]) - 1] = pd.DataFrame(json.load(f)["interactions"])
    return out
