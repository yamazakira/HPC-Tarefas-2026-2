"""
Tarefa 18 - Graficos do escalonador Lider-Trabalhador

Le todos os resultados_np<P>.json da pasta da atividade e salva:
    tarefa18_tempo.png          -> tempo total x numero de fontes (por nº de processos)
    tarefa18_eficiencia.png     -> eficiencia x numero de fontes
    tarefa18_distribuicao.png   -> tarefas e tempo ocupado por trabalhador (maior nº de fontes)
    tarefa18_gantt.png          -> linha do tempo dos tiros em cada trabalhador
"""
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = os.path.dirname(os.path.abspath(__file__))

arquivos = sorted(glob.glob(os.path.join(PASTA, "resultados_np*.json")),
                  key=lambda c: int(c.split("np")[-1].split(".")[0]))
if not arquivos:
    raise SystemExit("Nenhum resultados_np*.json encontrado.")

execucoes = []
for caminho in arquivos:
    with open(caminho, encoding="utf-8") as f:
        execucoes.append(json.load(f))

# ============================================================
# Grafico 1 - Tempo total x numero de fontes
# ============================================================
plt.figure(figsize=(11, 7))
for ex in execucoes:
    exps = ex["experimentos"]
    w = ex["trabalhadores"]
    ns = [e["n_fontes"] for e in exps]
    linha, = plt.plot(ns, [e["tempo_total_s"] for e in exps], marker="o",
                      linewidth=2, label=f"{ex['processos']} processos ({w} trabalhadores)")
    # Ideal: ceil(n / W) "ondas" de tiros, cada uma com o tempo medio de um tiro
    t_medio = sum(e["soma_tempo_tiros_s"] for e in exps) / sum(ns)
    plt.plot(ns, [math.ceil(n / w) * t_medio for n in ns], linestyle=":",
             color=linha.get_color(), alpha=0.8)

ns_ref = [e["n_fontes"] for e in execucoes[0]["experimentos"]]
t_seq = sum(e["soma_tempo_tiros_s"] for e in execucoes[0]["experimentos"]) / sum(ns_ref)
plt.plot(ns_ref, [n * t_seq for n in ns_ref], "k--", label="Sequencial estimado (n × t_tiro)")
plt.plot([], [], "k:", label="Ideal: ⌈n / W⌉ × t_tiro")

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Número de fontes sísmicas (tiros)")
plt.ylabel("Tempo total (s)")
plt.title("Líder-Trabalhador - tempo de execução x número de fontes")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa18_tempo.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 2 - Eficiencia x numero de fontes
# ============================================================
plt.figure(figsize=(11, 6))
for ex in execucoes:
    exps = ex["experimentos"]
    plt.plot([e["n_fontes"] for e in exps], [100 * e["eficiencia"] for e in exps],
             marker="o", linewidth=2,
             label=f"{ex['processos']} processos ({ex['trabalhadores']} trabalhadores)")
plt.xscale("log", base=2)
plt.ylim(0, 105)
plt.xlabel("Número de fontes sísmicas (tiros)")
plt.ylabel("Eficiência (%)  [Σ t_tiro / (T_total × W)]")
plt.title("Líder-Trabalhador - eficiência dos trabalhadores")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa18_eficiencia.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 3 - Distribuicao das tarefas (maior experimento de cada execucao)
# ============================================================
fig, eixos = plt.subplots(1, len(execucoes), figsize=(5 * len(execucoes), 5),
                          squeeze=False)
for ax, ex in zip(eixos[0], execucoes):
    e = ex["experimentos"][-1]
    ws = list(range(1, ex["trabalhadores"] + 1))
    ax.bar(ws, e["tarefas_por_trabalhador"], color="tab:blue", alpha=0.8)
    ax.set_xlabel("Trabalhador (rank)")
    ax.set_ylabel("Tiros processados", color="tab:blue")
    ax.set_xticks(ws)
    ax2 = ax.twinx()
    ax2.plot(ws, e["tempo_ocupado_por_trabalhador_s"], "o-", color="tab:red")
    ax2.set_ylabel("Tempo ocupado (s)", color="tab:red")
    ax2.set_ylim(0, max(e["tempo_ocupado_por_trabalhador_s"]) * 1.2)
    ax.set_title(f"{ex['processos']} processos - {e['n_fontes']} fontes")
fig.suptitle("Distribuição das tarefas entre os trabalhadores")
fig.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa18_distribuicao.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 4 - Gantt (maior nº de processos, experimento com 4W fontes ou o maior)
# ============================================================
ex = execucoes[-1]
w_total = ex["trabalhadores"]
alvo = next((e for e in ex["experimentos"] if e["n_fontes"] >= 4 * w_total),
            ex["experimentos"][-1])
cores = plt.get_cmap("tab20")

plt.figure(figsize=(12, max(4, 0.45 * w_total + 2)))
for t in alvo["tiros"]:
    plt.barh(t["trabalhador"], t["fim_s"] - t["inicio_s"], left=t["inicio_s"],
             color=cores(t["id"] % 20), edgecolor="black", linewidth=0.4)
    if alvo["n_fontes"] <= 64:
        plt.text(t["inicio_s"] + (t["fim_s"] - t["inicio_s"]) / 2, t["trabalhador"],
                 str(t["id"]), ha="center", va="center", fontsize=7)
plt.yticks(range(1, w_total + 1))
plt.xlabel("Tempo desde o início do experimento (s)")
plt.ylabel("Trabalhador (rank)")
plt.title(f"Linha do tempo dos tiros - {ex['processos']} processos, "
          f"{alvo['n_fontes']} fontes")
plt.grid(True, axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa18_gantt.png"), dpi=300)
plt.close()

print("Graficos salvos em", PASTA)
