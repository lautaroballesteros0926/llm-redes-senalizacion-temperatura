"""
Copia adaptada de LLM_prompting.py (repo saucermanlab/LLM-network-generation, licencia MIT)
para usar la API de DeepSeek (compatible con OpenAI) con el modelo deepseek-flash
(DeepSeek V4.1 Flash) con el razonamiento DESACTIVADO.

Importante para el experimento: en DeepSeek la temperatura NO tiene efecto en modo razonamiento
(que viene activado por defecto), por eso se envía thinking={"type": "disabled"} en cada llamada.

Se mantiene igual que el original
---------------------------------
* El prompt inicial y el de seguimiento (texto idéntico; hay una prueba que lo comprueba).
* La conversación única que crece: lista de genes -> lote 1 -> "That looks great!..." -> ...
* Lotes de 20 genes (batch_size = 20) y max_tokens = 4000.
* El formato de salida: source_node, target_node, action ("stimulation" | "inhibition").
* La lista de genes y max_connections salen del Excel de la red igual que en
  network_extraction.extract_network_data (pero SIN escribir archivos dentro del repo).

Lo que cambia (y por qué)
-------------------------
1. Proveedor: cliente OpenAI apuntando a DeepSeek en vez de GPT/Claude/Gemini.
2. DeepSeek solo ofrece el modo "json_object" (garantiza JSON válido pero no la forma; a veces
   devuelve contenido vacío). Los otros 3 LLMs usaban salida estructurada con esquema.
   Para compensar añadimos un mensaje de sistema que describe la forma esperada y
   validamos cada respuesta con el mismo modelo pydantic del repo.
3. En el historial guardamos la respuesta tal cual la devolvió el modelo (JSON), en vez de
   str(objeto_pydantic) como hacía el repo.
4. Reintentos, control de ritmo y guardado en disco de cada respuesta apenas llega.
"""
import sys
sys.dont_write_bytecode = True

import os, re, json, time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
import openai

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent / "LLM-network-generation"

CONFIG = {"base_url": "https://api.deepseek.com", "env": "DEEPSEEK_API_KEY", "modelo": "deepseek-flash",
          "extra_body": {"thinking": {"type": "disabled"}},   # sin esto la temperatura no tiene efecto
          "rpm": 600, "tpm": 10**9, "max_solicitud": 10**9}   # DeepSeek no limita RPM/TPM (solo concurrencia)
PROVEEDOR = "deepseek"
MODELO = CONFIG["modelo"]
BASE_URL = CONFIG["base_url"]
BATCH_SIZE = 20          # igual que el repo
MAX_TOKENS = 4000        # igual que el repo (--max_tokens por defecto)
# Si una respuesta se corta por longitud con 4000 tokens (pasa a temperatura 0: el modelo da listas más largas),
# el reintento usa este tope mayor. Cada intento queda registrado con el max_tokens que usó.
MAX_TOKENS_REINTENTO = 16000
DESC = {"hypertrophy": "cardiac hypertrophy"}   # descripción usada en el prompt (como en los archivos del repo)

SISTEMA = ('Respond only with a JSON object of the form {"interactions": [{"source_node": "...", '
           '"target_node": "...", "action": "stimulation" or "inhibition"}]}.')


# ---------- formato de salida (idéntico al repo) ----------
class GeneInteraction(BaseModel):
    source_node: str
    target_node: str
    action: Literal["stimulation", "inhibition"]


class InteractionList(BaseModel):
    interactions: list[GeneInteraction]


class LimiteDiarioAlcanzado(Exception):
    """Se acabó el saldo (HTTP 402) o el gasto está bloqueado: hay que parar y reanudar después."""


class SolicitudMuyGrande(Exception):
    """La solicitud no cabe en el límite de tokens por minuto."""


# ---------- genes de la red (copia de network_extraction.extract_network_data, sin escribir archivos) ----------
def obtener_genes(red):
    ruta = REPO / "models" / f"{red}.xlsx"
    df_species = pd.read_excel(ruta, skiprows=1, sheet_name="species")
    df_reactions = pd.read_excel(ruta, skiprows=1, sheet_name="reactions")

    def limpiar(texto):
        return re.sub(r"[^a-zA-Z0-9]", "", texto)

    conectividad = {}
    for _, row in df_reactions.iterrows():
        nodos = [limpiar(n) for n in row["Rule"].split(" ")]
        nodos = [n for n in nodos if n and n.strip()]
        for n in nodos:
            conectividad[n] = conectividad.get(n, 0) + 1
    max_conexiones = max(conectividad.values())

    genes = []
    for gene_name in df_species["gene_name"].tolist():
        if pd.isna(gene_name):
            continue
        elif ";" in gene_name:
            genes.extend(gene_name.split(";"))
        elif gene_name != "NaN":
            genes.append(gene_name)
    return max_conexiones, genes


# ---------- prompts (texto idéntico al de LLM_prompting.py, incluidos los espacios) ----------
def prompt_inicial(desc, max_connections, batch_size=BATCH_SIZE):
    return (f"For the first {batch_size} entries in this list of genes, proteins, and other signaling nodes \n"
            f"    from a {desc} signaling network, please provide more than 0 but fewer than {max_connections} direct interactions \n"
            f"    with other nodes from the list supported by available literature. \n"
            f"    Return plausible interactions between these nodes in JSON format. \n"
            f'    Each interaction should have: source_node, target_node, and action ("stimulation" or "inhibition").\n'
            f"    ")


def prompt_seguimiento(n_chunk):
    return f"That looks great! Please do the same operation for the next {n_chunk} nodes! Thank you"


def mensaje_genes(genes):
    return "List of genes and other signaling nodes:" + str(genes)


# ---------- cliente ----------
def crear_cliente():
    load_dotenv(AQUI / ".env")
    var = CONFIG["env"]
    key = (os.getenv(var) or "").strip()
    if not key or key.upper().startswith("PEGA"):
        raise RuntimeError(f"Falta la API key: pégala en experimento/.env como {var}=... (sin comillas).")
    # max_retries=0: los reintentos los manejamos nosotros para poder registrarlos
    return openai.OpenAI(api_key=key, base_url=BASE_URL, max_retries=0, timeout=120)


def listar_modelos():
    """Modelos que ve la cuenta (no gasta tokens)."""
    return sorted(m.id for m in crear_cliente().models.list().data)


# ---------- control de ritmo (límites reportados: 30 RPM, 12 000 TPM; usamos margen) ----------
class Ritmo:
    def __init__(self, salida_tipica=2500):
        cfg = CONFIG
        self.rpm, self.tpm, self.max_solicitud, self.salida_tipica = cfg["rpm"], cfg["tpm"], cfg["max_solicitud"], salida_tipica
        self.llamadas = deque()   # instantes
        self.tokens = deque()     # (instante, tokens)

    def _purgar(self):
        corte = time.time() - 60
        while self.llamadas and self.llamadas[0] < corte:
            self.llamadas.popleft()
        while self.tokens and self.tokens[0][0] < corte:
            self.tokens.popleft()

    def estimar(self, messages):
        chars = sum(len(m["content"]) for m in messages)
        return int(chars / 3) + self.salida_tipica

    def esperar(self, messages):
        est = self.estimar(messages)
        if est > self.max_solicitud:
            raise SolicitudMuyGrande(f"Solicitud estimada en ~{est} tokens: no cabe en el límite por minuto ({self.max_solicitud}).")
        while True:
            self._purgar()
            if len(self.llamadas) < self.rpm and sum(t for _, t in self.tokens) + est <= self.tpm:
                return
            time.sleep(2)

    def registrar(self, tokens_reales):
        self.llamadas.append(time.time())
        self.tokens.append((time.time(), tokens_reales))


# ---------- interpretar la respuesta ----------
def parsear(contenido):
    """-> (interacciones_validas, n_descartadas, normalizada, error)"""
    texto = (contenido or "").strip()
    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*", "", texto, flags=re.IGNORECASE)
        texto = re.sub(r"\s*```$", "", texto)
    try:
        data = json.loads(texto)
    except Exception as e:
        return [], 0, False, f"JSON inválido: {e}"
    normalizada = False
    if isinstance(data, list):
        items, normalizada = data, True
    elif isinstance(data, dict) and isinstance(data.get("interactions"), list):
        items = data["interactions"]
    elif isinstance(data, dict):
        listas = [v for v in data.values() if isinstance(v, list)]
        if len(listas) != 1:
            return [], 0, False, "JSON sin lista 'interactions'"
        items, normalizada = listas[0], True
    else:
        return [], 0, False, "JSON de forma inesperada"
    validas, descartadas = [], 0
    for it in items:
        try:
            validas.append(GeneInteraction.model_validate(it).model_dump())
        except Exception:
            descartadas += 1
    return validas, descartadas, normalizada, None


# ---------- una llamada a la API con reintentos ----------
def _cabeceras_limite(headers):
    return {k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"}


def llamar_api(client, messages, temperatura, max_tokens=MAX_TOKENS, max_intentos=6):
    """
    Devuelve dict con: content, finish_reason, usage, headers, intentos (lista de incidencias).
    Reintenta con espera ante 429 (por minuto), errores 5xx y de conexión. Lanza LimiteDiarioAlcanzado
    si el 429 es por cuota diaria y SolicitudMuyGrande si es un 413.
    Un 400 (p. ej. json_validate_failed) se devuelve como estado 'rechazada' sin reintentar aquí.
    """
    incidencias = []
    for intento in range(1, max_intentos + 1):
        try:
            crudo = client.chat.completions.with_raw_response.create(
                model=MODELO, messages=messages, temperature=temperatura, max_tokens=max_tokens,
                response_format={"type": "json_object"}, extra_body=CONFIG["extra_body"])
            comp = crudo.parse()
            choice = comp.choices[0]
            razonamiento = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
            return {"content": choice.message.content, "finish_reason": choice.finish_reason,
                    "refusal": getattr(choice.message, "refusal", None),
                    "hubo_razonamiento": bool(razonamiento),   # debe ser False: si no, la temperatura no tendría efecto
                    "usage": comp.usage.model_dump() if comp.usage else None,
                    "headers": _cabeceras_limite(crudo.headers), "modelo_devuelto": comp.model,
                    "incidencias": incidencias, "error_400": None}
        except openai.RateLimitError as e:
            msg = str(e)
            hdr = _cabeceras_limite(e.response.headers) if e.response is not None else {}
            incidencias.append({"intento": intento, "tipo": "429", "msg": msg[:300], "headers": hdr})
            if re.search(r"per day|\bTPD\b|\bRPD\b", msg, flags=re.IGNORECASE):
                raise LimiteDiarioAlcanzado(msg)
            try:
                espera = float(hdr.get("retry-after", 0))
            except ValueError:
                espera = 0
            time.sleep(max(espera, min(5 * 2 ** (intento - 1), 60)) + 1)
        except openai.APIStatusError as e:
            incidencias.append({"intento": intento, "tipo": str(e.status_code), "msg": str(e)[:300]})
            if e.status_code == 413:
                raise SolicitudMuyGrande(str(e))
            if e.status_code == 402 or (e.status_code == 400 and "blocked_api_access" in str(e)):
                raise LimiteDiarioAlcanzado(f"Sin saldo o gasto bloqueado: {str(e)[:200]}")
            if e.status_code == 400:
                return {"content": None, "finish_reason": None, "refusal": None, "usage": None, "headers": {}, "hubo_razonamiento": False,
                        "modelo_devuelto": None, "incidencias": incidencias, "error_400": str(e)[:600]}
            if e.status_code >= 500:
                time.sleep(min(5 * 2 ** (intento - 1), 60))
            else:
                raise
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            incidencias.append({"intento": intento, "tipo": "conexion", "msg": str(e)[:300]})
            time.sleep(min(5 * 2 ** (intento - 1), 60))
    raise RuntimeError(f"La API no respondió tras {max_intentos} intentos: {incidencias[-1]}")


def _guardar(ruta, obj):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ruta)   # escritura atómica: nunca queda un archivo a medias


def _estado(info, validas, error):
    if info["error_400"]:
        return "rechazada"
    if info["refusal"] or info["finish_reason"] == "content_filter":
        return "rechazada"
    if not (info["content"] or "").strip():
        return "sin_contenido"          # la API devolvió texto vacío (DeepSeek lo hace a veces en modo JSON): se reintenta
    if error:
        # JSON incompleto porque la respuesta chocó con el límite de tokens de salida
        return "cortada" if info["finish_reason"] == "length" else "malformada"
    if not validas:
        return "vacia"
    return "ok"


# ---------- una réplica completa (todas las llamadas de una conversación) ----------
def generar_replica(client, genes, max_conexiones, desc, temperatura, carpeta, ritmo,
                    max_intentos_malformada=3, verbose=True):
    """
    Guarda cada respuesta en carpeta/llamada_NN.json apenas llega. Si la carpeta ya tiene llamadas
    aceptadas (ok/vacia), se reutilizan: así se puede reanudar tras un corte sin repetir ni gastar cuota.
    Devuelve {'interactions': [...], 'completa': bool, 'llamadas': [registro, ...]}.
    """
    carpeta = Path(carpeta)
    historial = [{"role": "system", "content": SISTEMA},
                 {"role": "user", "content": mensaje_genes(genes)}]
    prompts = [prompt_inicial(desc, max_conexiones)]
    for i in range(BATCH_SIZE, len(genes), BATCH_SIZE):
        prompts.append(prompt_seguimiento(len(genes[i:i + BATCH_SIZE])))

    todas, registros, completa = [], [], True
    for k, prompt in enumerate(prompts, start=1):
        historial.append({"role": "user", "content": prompt})
        archivo = carpeta / f"llamada_{k:02d}.json"
        previo = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else None

        if previo and previo["estado"] in ("ok", "vacia"):
            reg = previo                                    # reutilizar respuesta ya guardada
            reg["reutilizada"] = True
        else:
            intentos, max_tok = [], MAX_TOKENS
            if previo:   # se reanuda una llamada que había fallado: conservar el registro de sus intentos anteriores
                intentos = list(previo.get("intentos_previos", [])) + [
                    {"intento": previo.get("intento"), "estado": previo["estado"], "error": previo.get("error"),
                     "finish_reason": previo.get("finish_reason"), "max_tokens": previo.get("max_tokens_usado", MAX_TOKENS)}]
                if previo.get("finish_reason") == "length":
                    max_tok = MAX_TOKENS_REINTENTO
            for n in range(1, max_intentos_malformada + 1):
                ritmo.esperar(historial)
                t0 = time.time()
                info = llamar_api(client, historial, temperatura, max_tokens=max_tok)
                ritmo.registrar((info["usage"] or {}).get("total_tokens", ritmo.estimar(historial)))
                validas, descartadas, normalizada, error = parsear(info["content"]) if not info["error_400"] else ([], 0, False, info["error_400"])
                estado = _estado(info, validas, error)
                reg = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "modelo": MODELO,
                       "temperatura_enviada": temperatura, "llamada": k, "intento": len(intentos) + 1, "max_tokens_usado": max_tok,
                       "prompt_usuario": prompt, "content": info["content"], "finish_reason": info["finish_reason"],
                       "usage": info["usage"], "headers_limite": info["headers"], "incidencias_api": info["incidencias"],
                       "hubo_razonamiento": info["hubo_razonamiento"],
                       "error": error, "n_interacciones": len(validas), "n_descartadas": descartadas,
                       "normalizada": normalizada, "estado": estado, "segundos": round(time.time() - t0, 2),
                       "interacciones": validas, "reutilizada": False, "intentos_previos": intentos}
                _guardar(archivo, reg)                       # a disco apenas llega
                if estado in ("ok", "vacia"):
                    break
                intentos.append({"intento": len(intentos) + 1, "estado": estado, "error": error,
                                 "finish_reason": info["finish_reason"], "max_tokens": max_tok})
                if verbose:
                    print(f"   llamada {k}: {estado} (finish={info['finish_reason']}, max_tokens={max_tok}); intento {n}/{max_intentos_malformada}")
                if info["finish_reason"] == "length":
                    max_tok = MAX_TOKENS_REINTENTO           # el siguiente intento deja más espacio de salida
        registros.append({c: reg[c] for c in ("llamada", "estado", "n_interacciones", "n_descartadas", "normalizada",
                                              "finish_reason", "usage", "segundos", "reutilizada", "intentos_previos")
                          if c in reg})
        if reg["estado"] not in ("ok", "vacia"):
            completa = False
            break                                            # réplica incompleta: se puede reanudar después
        historial.append({"role": "assistant", "content": reg["content"] or "{}"})
        todas.extend(reg["interacciones"])
        if verbose:
            tok = (reg.get("usage") or {}).get("total_tokens")
            print(f"   llamada {k}/{len(prompts)}: {reg['estado']}, {reg['n_interacciones']} interacciones"
                  + (f", {tok} tokens" if tok else "") + (" (reutilizada)" if reg.get("reutilizada") else ""))

    resultado = {"interactions": todas}
    if completa:
        _guardar(carpeta / "replica.json", resultado)       # mismo formato que los JSON del repo
    return {"interactions": todas, "completa": completa, "llamadas": registros}


# ---------- una sola llamada de prueba (Etapa 4) ----------
def llamada_de_prueba(red="hypertrophy", temperatura=1.0, carpeta=None):
    carpeta = Path(carpeta or AQUI / "resultados" / "prueba")
    max_conexiones, genes = obtener_genes(red)
    client = crear_cliente()
    historial = [{"role": "system", "content": SISTEMA},
                 {"role": "user", "content": mensaje_genes(genes)},
                 {"role": "user", "content": prompt_inicial(DESC[red], max_conexiones)}]
    ritmo = Ritmo()
    ritmo.esperar(historial)
    t0 = time.time()
    info = llamar_api(client, historial, temperatura)
    validas, descartadas, normalizada, error = parsear(info["content"]) if not info["error_400"] else ([], 0, False, info["error_400"])
    resumen = {"proveedor": PROVEEDOR, "modelo_pedido": MODELO, "modelo_devuelto": info["modelo_devuelto"],
               "razonamiento_desactivado_en_peticion": CONFIG["extra_body"],
               "hubo_razonamiento_en_respuesta": info["hubo_razonamiento"], "temperatura_enviada": temperatura,
               "finish_reason": info["finish_reason"], "json_valido": error is None, "error": error,
               "n_interacciones_validas": len(validas), "n_descartadas": descartadas, "forma_alternativa": normalizada,
               "usage": info["usage"], "headers_limite": info["headers"], "incidencias_api": info["incidencias"],
               "segundos": round(time.time() - t0, 2), "estado": _estado(info, validas, error),
               "n_genes": len(genes), "max_conexiones": max_conexiones, "llamadas_por_replica": 1 + len(range(BATCH_SIZE, len(genes), BATCH_SIZE))}
    _guardar(carpeta / "prueba_respuesta.json", {**resumen, "content": info["content"], "interacciones": validas})
    return resumen
