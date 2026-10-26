"""
Ejecuta el experimento de la Etapa 5: red de hipertrofia, DeepSeek V4.1 Flash (razonamiento desactivado).

  * Temperatura 1.0 -> 10 réplicas      * Temperatura 0 -> 3 réplicas
  * 12 llamadas por réplica (225 genes en lotes de 20), con el historial de conversación como en el repo.

Las réplicas se intercalan (T1.0 #1, T0 #1, T1.0 #2, ...) para que ninguna condición quede en otra hora del día.
Cada respuesta se guarda en disco apenas llega (resultados/deepseek/T<temp>/repNN/llamada_NN.json) y las
réplicas ya completas (replica.json) se saltan, así que se puede volver a lanzar sin repetir ni gastar de más.

Uso:  python -u ejecutar_experimento.py
"""
import sys
sys.dont_write_bytecode = True
import os, json, time
from datetime import datetime, timezone
from pathlib import Path
import requests

import llm_prompting_deepseek as gq

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "resultados" / "deepseek"
RED = "hypertrophy"
PLAN = [(1.0, i) for i in range(1, 11)] + [(0.0, i) for i in range(1, 4)]


def orden_intercalado():
    out = []
    for i in range(1, 11):
        out.append((1.0, i))
        if i <= 3:
            out.append((0.0, i))
    return out


def saldo():
    try:
        r = requests.get("https://api.deepseek.com/user/balance", timeout=30,
                         headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"], "Accept": "application/json"})
        return r.json()["balance_infos"][0]["total_balance"]
    except Exception as e:
        return f"?({type(e).__name__})"


def carpeta(temp, rep):
    return SALIDA / f"T{temp:.1f}" / f"rep{rep:02d}"


def registrar(linea):
    SALIDA.mkdir(parents=True, exist_ok=True)
    with open(SALIDA / "registro_ejecucion.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(linea, ensure_ascii=False) + "\n")


def main():
    client = gq.crear_cliente()
    max_conexiones, genes = gq.obtener_genes(RED)
    desc = gq.DESC[RED]
    ritmo = gq.Ritmo()
    print(f"[{datetime.now():%H:%M:%S}] Modelo {gq.MODELO} | {len(genes)} genes | max_connections={max_conexiones} | saldo inicial USD {saldo()}", flush=True)
    registrar({"evento": "inicio", "utc": datetime.now(timezone.utc).isoformat(), "modelo": gq.MODELO,
               "saldo": saldo(), "n_genes": len(genes), "max_conexiones": max_conexiones})

    pendientes = orden_intercalado()
    for pasada in (1, 2):                      # la 2.ª pasada solo reintenta réplicas que quedaron incompletas
        incompletas = []
        for temp, rep in pendientes:
            c = carpeta(temp, rep)
            if (c / "replica.json").exists():
                print(f"[{datetime.now():%H:%M:%S}] T={temp} rep{rep:02d}: ya completa, se salta", flush=True)
                continue
            print(f"[{datetime.now():%H:%M:%S}] === T={temp} rep{rep:02d} (pasada {pasada}) ===", flush=True)
            t0 = time.time()
            try:
                r = gq.generar_replica(client, genes, max_conexiones, desc, temp, c, ritmo)
            except gq.LimiteDiarioAlcanzado as e:
                print(f"\n!!! PARADA: {e}\nReanuda volviendo a lanzar este script.", flush=True)
                registrar({"evento": "parada_saldo", "utc": datetime.now(timezone.utc).isoformat(), "detalle": str(e)[:300]})
                return 2
            tokens = sum((x.get("usage") or {}).get("total_tokens", 0) for x in r["llamadas"] if not x.get("reutilizada"))
            registrar({"evento": "replica", "temperatura": temp, "replica": rep, "pasada": pasada, "completa": r["completa"],
                       "n_interacciones": len(r["interactions"]), "tokens_llamadas_nuevas": tokens,
                       "segundos": round(time.time() - t0, 1), "saldo_despues": saldo(),
                       "estados": [x["estado"] for x in r["llamadas"]]})
            print(f"[{datetime.now():%H:%M:%S}] T={temp} rep{rep:02d}: {'COMPLETA' if r['completa'] else 'INCOMPLETA'}, "
                  f"{len(r['interactions'])} interacciones, {time.time()-t0:.0f}s, saldo USD {saldo()}", flush=True)
            if not r["completa"]:
                incompletas.append((temp, rep))
        pendientes = incompletas
        if not pendientes:
            break
    faltan = [(t, r) for t, r in PLAN if not (carpeta(t, r) / "replica.json").exists()]
    print(f"\n[{datetime.now():%H:%M:%S}] FIN. Réplicas sin completar: {faltan or 'ninguna'} | saldo final USD {saldo()}", flush=True)
    registrar({"evento": "fin", "utc": datetime.now(timezone.utc).isoformat(), "sin_completar": faltan, "saldo": saldo()})
    return 0 if not faltan else 1


if __name__ == "__main__":
    sys.exit(main())
