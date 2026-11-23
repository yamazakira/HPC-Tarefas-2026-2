import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = os.path.dirname(os.path.abspath(__file__))

NOMES = {
    "false": "Sem afinidade (SO decide)",
    "taskset": "SO: taskset -c 0..p-1",
    "close": "OMP_PROC_BIND=close",
    "spread": "OMP_PROC_BIND=spread",
    "master": "OMP_PROC_BIND=master",
}

dados = defaultdict(dict)
with open(os.path.join(PASTA, "resultados.out"), encoding="utf-8") as f:
    for linha in f:
        if linha.strip():
            r = json.loads(linha)
            exp, _, af = r["experimento"].partition(":")
            dados[(exp, af or "padrao")][r["threads"]] = r

afinidades = sorted({af for _, af in dados}, key=lambda a: list(NOMES).index(a) if a in NOMES else 99)


def curva(exp, af):
    pts = dados.get((exp, af), {})
    ts = sorted(pts)
    return ts, [pts[t]["tempo_s"] for t in ts], pts


def grafico(exp, metrica, arquivo, titulo, ylabel):
    plt.figure(figsize=(11, 7))
    tmax = 1
    for af in afinidades:
        ts, tempos, pts = curva(exp, af)
        if not ts or 1 not in pts:
            continue
        t1 = pts[1]["tempo_s"]
        tmax = max(tmax, max(ts))
        if metrica == "speedup":
            ys = [t1 / tp for tp in tempos]
        elif metrica == "speedup_fraca":
            ys = [t * t1 / tp for tp, t in zip(tempos, ts)]
        elif metrica == "eficiencia":
            ys = [100 * t1 / tp / t for tp, t in zip(tempos, ts)]
        else:
            ys = [100 * t1 / tp for tp in tempos]
        plt.plot(ts, ys, marker="o", linewidth=2, label=NOMES.get(af, af))
    if metrica in ("speedup", "speedup_fraca"):
        plt.plot([1, tmax], [1, tmax], "k--", label="Ideal")
    else:
        plt.axhline(100, color="black", linestyle="--", label="Ideal")
    plt.xticks(sorted({t for (e, _), v in dados.items() if e == exp for t in v}))
    plt.xlabel("Número de threads")
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA, arquivo), dpi=300)
    plt.close()


exemplo = next(iter(next(iter(dados.values())).values()))
forte = next((p for (e, _), v in dados.items() if e == "forte" for p in v.values()), exemplo)

grafico("forte", "speedup", "tarefa12_speedup.png",
        f"Escalabilidade forte - speedup (malha {forte['nx']}x{forte['nz']}, {forte['nt']} passos)",
        "Speedup (T1 / Tp)")
grafico("forte", "eficiencia", "tarefa12_eficiencia.png",
        f"Escalabilidade forte - eficiência (malha {forte['nx']}x{forte['nz']})",
        "Eficiência (%)  [T1 / (p · Tp)]")
grafico("fraca", "speedup_fraca", "tarefa12_speedup_fraca.png",
        "Escalabilidade fraca - speedup escalado (malha cresce com o nº de threads)",
        "Speedup escalado (p · T1 / Tp)")
grafico("fraca", "fraca", "tarefa12_fraca.png",
        "Escalabilidade fraca - eficiência (malha cresce com o nº de threads)",
        "Eficiência fraca (%)  [T1 / Tp]")

plt.figure(figsize=(11, 7))
for exp, estilo in (("forte", "-"), ("fraca", "--")):
    for af in afinidades:
        ts, tempos, _ = curva(exp, af)
        if ts:
            plt.plot(ts, tempos, marker="o", linestyle=estilo,
                     label=f"{exp} - {NOMES.get(af, af)}")
plt.yscale("log")
plt.xlabel("Número de threads")
plt.ylabel("Tempo (s)")
plt.title("Tempo de execução por política de afinidade")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa12_tempos.png"), dpi=300)
plt.close()

print("Graficos salvos em", PASTA)
