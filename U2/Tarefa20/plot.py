"""
Tarefa 20 - Graficos da integral por trapezios (MPI + OpenMP + GPU)

Le todos os resultados_np<P>.json da pasta da atividade e salva:
    tarefa20_tempo.png        -> tempo total x N (CPU, GPU) para cada nº de processos
    tarefa20_escala.png       -> speedup x nº de processos (maior N)
    tarefa20_comunicacao.png  -> custo do envio periodico ao vizinho (cpu x cpu_sem_backup)
    tarefa20_erro.png         -> erro relativo x N (validacao contra pi)
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))

arquivos = glob.glob(os.path.join(PASTA, "resultados*np*.json"))
if not arquivos:
    raise SystemExit("Nenhum resultados_np*.json encontrado.")

execucoes = []
for caminho in arquivos:
    with open(caminho, encoding="utf-8") as f:
        execucoes.append(json.load(f))
execucoes.sort(key=lambda e: (e.get("rotulo", ""), e["processos"]))

ps = sorted({e["processos"] for e in execucoes})
rotulos = sorted({e.get("rotulo", "") for e in execucoes})
NOMES = {"cpu": "CPU (com backup)", "cpu_sem_backup": "CPU (sem backup)",
         "gpu": "GPU (com backup)"}


def nos_distintos(ex):
    return len({n["no"] for n in ex["nos"]})


def serie(ex, modo):
    pts = sorted((r for r in ex["resultados"] if r["modo"] == modo),
                 key=lambda r: r["n"])
    return [r["n"] for r in pts], pts


def rotulo(ex):
    prefixo = f"[{ex['rotulo']}] " if ex.get("rotulo") else ""
    return (f"{prefixo}P={ex['processos']} ({nos_distintos(ex)} nó(s), "
            f"{ex['threads']} threads)")


# ============================================================
# Grafico 1 - Tempo x N
# ============================================================
plt.figure(figsize=(12, 7))
estilos = {"cpu": "-", "gpu": "--"}
for ex in execucoes:
    cor = None
    for modo in ("cpu", "gpu"):
        ns, pts = serie(ex, modo)
        if not pts:
            continue
        linha, = plt.plot(ns, [p["tempo_total_s"] for p in pts], marker="o",
                          linestyle=estilos[modo], linewidth=2, color=cor,
                          label=f"{rotulo(ex)} - {NOMES[modo]}")
        cor = linha.get_color()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Número de trapézios (N)")
plt.ylabel("Tempo total (s)  [cálculo + MPI_Reduce]")
plt.title("Integral por trapézios - tempo de execução")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa20_tempo.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 2 - Speedup x processos (maior N)
# ============================================================
n_max = max(r["n"] for r in execucoes[0]["resultados"])
plt.figure(figsize=(10, 6))
for rot in rotulos:
    grupo = [e for e in execucoes if e.get("rotulo", "") == rot]
    for modo in ("cpu", "cpu_sem_backup", "gpu"):
        xs, ys = [], []
        for ex in grupo:
            t = [r["tempo_total_s"] for r in ex["resultados"]
                 if r["modo"] == modo and r["n"] == n_max]
            if t:
                xs.append(ex["processos"])
                ys.append(t[0])
        if xs:
            nome = (f"[{rot}] " if rot else "") + NOMES[modo]
            plt.plot(xs, [ys[0] / y for y in ys], marker="o",
                     linewidth=2, label=nome)
plt.plot(ps, [p / ps[0] for p in ps], "k--", label="Ideal")
plt.ylabel("Speedup em relação ao menor P de cada série")
plt.xscale("log", base=2)
plt.xticks(ps, [str(p) for p in ps])
plt.xlabel("Número de processos MPI")
plt.title(f"Escalabilidade entre processos/nós (N = {n_max:.0e})")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa20_escala.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 3 - Impacto da comunicacao (backup ao vizinho)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
largura = 0.8 / len(execucoes)
ns_todos = sorted({r["n"] for r in execucoes[0]["resultados"]})
for k, ex in enumerate(execucoes):
    com = {r["n"]: r for r in ex["resultados"] if r["modo"] == "cpu"}
    sem = {r["n"]: r for r in ex["resultados"] if r["modo"] == "cpu_sem_backup"}
    sobrecarga = [100 * (com[n]["tempo_total_s"] - sem[n]["tempo_total_s"])
                  / sem[n]["tempo_total_s"] for n in ns_todos]
    msgs = [com[n]["mensagens_backup"] for n in ns_todos]
    pos = np.arange(len(ns_todos)) + k * largura
    ax1.bar(pos, sobrecarga, largura, label=rotulo(ex))
    ax2.bar(pos, msgs, largura, label=rotulo(ex))
for ax in (ax1, ax2):
    ax.set_xticks(np.arange(len(ns_todos)) + largura * (len(execucoes) - 1) / 2)
    ax.set_xticklabels([f"{n:.0e}" for n in ns_todos])
    ax.set_xlabel("Número de trapézios (N)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
ax1.axhline(0, color="black", linewidth=1)
ax1.set_ylabel("Sobrecarga do backup (%)  [(T_com − T_sem) / T_sem]")
ax1.set_title("Custo do envio periódico da soma parcial")
ax2.set_yscale("log")
ax2.set_ylabel("Mensagens de backup enviadas (total)")
ax2.set_title("Mensagens trocadas entre vizinhos")
fig.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa20_comunicacao.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 4 - Erro relativo x N
# ============================================================
plt.figure(figsize=(10, 6))
for ex in execucoes:
    ns, pts = serie(ex, "cpu")
    plt.plot(ns, [max(p["erro_rel"], 1e-17) for p in pts], marker="o",
             linewidth=2, label=rotulo(ex))
ns_ref = np.array(ns_todos, dtype=float)
# erro teorico do trapezio composto: (b-a) h^2 |f''| / 12, com |f''| medio ~ 2 -> ordem h^2
plt.plot(ns_ref, (1.0 / ns_ref) ** 2 / 6 / np.pi, "k--", label="Ordem teórica O(h²)")
plt.axhline(np.finfo(float).eps, color="gray", linestyle=":",
            label="Épsilon da máquina (double)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Número de trapézios (N)")
plt.ylabel("Erro relativo |I − π| / π")
plt.title("Validação: erro em relação ao valor analítico (π)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa20_erro.png"), dpi=300)
plt.close()

print("Graficos salvos em", PASTA)
