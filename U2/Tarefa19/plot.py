"""
Tarefa 19 - Graficos do produto matriz-vetor com MPI (Scatter/Bcast/Gather)

Le todos os resultados_np<P>.json da pasta da atividade e salva:
    tarefa19_tempo.png        -> tempo total x numero de processos (por tamanho)
    tarefa19_speedup.png      -> speedup total e speedup so do calculo x processos
    tarefa19_eficiencia.png   -> eficiencia x processos
    tarefa19_fases.png        -> tempo de cada fase (maior matriz) x processos
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))

arquivos = glob.glob(os.path.join(PASTA, "resultados_np*.json"))
if not arquivos:
    raise SystemExit("Nenhum resultados_np*.json encontrado.")

execucoes = []
for caminho in arquivos:
    with open(caminho, encoding="utf-8") as f:
        execucoes.append(json.load(f))
execucoes.sort(key=lambda e: e["processos"])

ps = [e["processos"] for e in execucoes]
nos = {e["processos"]: e["nos"] for e in execucoes}
tamanhos = sorted({r["n"] for e in execucoes for r in e["resultados"]})


def valor(p, n, chave):
    for e in execucoes:
        if e["processos"] == p:
            for r in e["resultados"]:
                if r["n"] == n:
                    return r[chave]
    return None


def serie(n, chave):
    xs, ys = [], []
    for p in ps:
        v = valor(p, n, chave)
        if v is not None:
            xs.append(p)
            ys.append(v)
    return xs, ys


rotulo_p = [f"{p}\n({nos[p]} nó{'s' if nos[p] > 1 else ''})" for p in ps]

# ============================================================
# Grafico 1 - Tempo total x processos
# ============================================================
plt.figure(figsize=(11, 7))
for n in tamanhos:
    xs, ys = serie(n, "tempo_total_s")
    plt.plot(xs, ys, marker="o", linewidth=2, label=f"{n}×{n}")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xticks(ps, rotulo_p)
plt.xlabel("Número de processos MPI")
plt.ylabel("Tempo total (s)  [Scatter + Bcast + cálculo + Gather]")
plt.title("y = A·x - tempo total x número de processos")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(title="Matriz")
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa19_tempo.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 2 - Speedup (total e somente calculo)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
for n in tamanhos:
    xs, ys = serie(n, "speedup")
    ax1.plot(xs, ys, marker="o", linewidth=2, label=f"{n}×{n}")
    xs, ys = serie(n, "speedup_calculo")
    ax2.plot(xs, ys, marker="o", linewidth=2, label=f"{n}×{n}")
for ax, titulo in ((ax1, "Speedup total (inclui comunicação)"),
                   (ax2, "Speedup apenas do cálculo local")):
    ax.plot(ps, ps, "k--", label="Ideal (S = P)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels(rotulo_p)
    ax.set_xlabel("Número de processos MPI")
    ax.set_ylabel("Speedup  (T_seq / T_P)")
    ax.set_title(titulo)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(title="Matriz", fontsize=8)
fig.suptitle("y = A·x - speedup em relação à versão sequencial")
fig.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa19_speedup.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 3 - Eficiencia
# ============================================================
plt.figure(figsize=(11, 6))
for n in tamanhos:
    xs, ys = serie(n, "eficiencia")
    plt.plot(xs, [100 * y for y in ys], marker="o", linewidth=2, label=f"{n}×{n}")
plt.axhline(100, color="black", linestyle="--", linewidth=1)
plt.xscale("log", base=2)
plt.xticks(ps, rotulo_p)
plt.xlabel("Número de processos MPI")
plt.ylabel("Eficiência (%)  [speedup / P]")
plt.title("y = A·x - eficiência paralela")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(title="Matriz")
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa19_eficiencia.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 4 - Fases (maior matriz presente em todas as execucoes)
# ============================================================
n_ref = max(n for n in tamanhos if all(valor(p, n, "tempo_total_s") is not None for p in ps))
fases = [("tempo_scatter_s", "MPI_Scatter (A)"), ("tempo_bcast_s", "MPI_Bcast (x)"),
         ("tempo_calculo_s", "Cálculo local"), ("tempo_gather_s", "MPI_Gather (y)")]
base = np.zeros(len(ps))
plt.figure(figsize=(11, 6))
for chave, nome in fases:
    vals = np.array([valor(p, n_ref, chave) for p in ps])
    plt.bar(range(len(ps)), vals, bottom=base, label=nome)
    base += vals
seq = valor(ps[0], n_ref, "tempo_sequencial_s")
plt.axhline(seq, color="black", linestyle="--", label=f"Sequencial ({seq:.4f} s)")
plt.xticks(range(len(ps)), rotulo_p)
plt.xlabel("Número de processos MPI")
plt.ylabel("Tempo (s)")
plt.title(f"y = A·x - tempo por fase (matriz {n_ref}×{n_ref})")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa19_fases.png"), dpi=300)
plt.close()

print("Graficos salvos em", PASTA)
