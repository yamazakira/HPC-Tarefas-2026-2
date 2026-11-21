"""
Tarefa 16 - Graficos da soma de vetores par/impar em CUDA

Le resultados_cuda.json (gerado por kernel.cu) e salva na pasta da atividade:
    tarefa16_tempos.png        -> tempo x N (CPU, GPU em varias configuracoes, transferencias)
    tarefa16_banda.png         -> largura de banda efetiva x N
    tarefa16_blocos.png        -> tempo e ocupacao teorica x tamanho do bloco (maior N)
    tarefa16_divergencia.png   -> kernel com desvio x kernel sem desvio
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PASTA, "resultados_cuda.json"), encoding="utf-8") as f:
    dados = json.load(f)

res = dados["resultados"]
base = dados["base"]
ns = sorted({r["n"] for r in res})
titulo_gpu = (f"GPU: {dados['gpu']} ({dados['sms']} SMs, "
              f"{dados['max_threads_por_sm']} threads/SM)")


def serie(config, kernel, bloco=None):
    pts = [r for r in res if r["config"] == config and r["kernel"] == kernel
           and (bloco is None or r["bloco"] == bloco)]
    pts.sort(key=lambda r: r["n"])
    return pts


series = [
    ("Bloco 32 (1 thread/elemento)", serie("uma_thread_por_elemento", "sem_desvio", 32)),
    ("Bloco 256 (1 thread/elemento)", serie("uma_thread_por_elemento", "sem_desvio", 256)),
    ("Bloco 1024 (1 thread/elemento)", serie("uma_thread_por_elemento", "sem_desvio", 1024)),
    (f"Propriedades do dispositivo (bloco {dados['bloco_api_ocupacao']}, "
     f"grid ≤ {dados['grid_persistente']}, grid-stride)",
     serie("dispositivo_grid_stride", "sem_desvio")),
    ("1 bloco por SM (subocupação)", serie("um_bloco_por_sm", "sem_desvio")),
]

# ============================================================
# Grafico 1 - Tempo x N
# ============================================================
plt.figure(figsize=(12, 7))
for nome, pts in series:
    plt.plot([p["n"] for p in pts], [p["tempo_kernel_ms"] for p in pts],
             marker="o", linewidth=2, label=nome)

plt.plot([b["n"] for b in base], [b["tempo_cpu_ms"] for b in base],
         marker="s", linestyle="--", color="black", label="CPU sequencial")
plt.plot([b["n"] for b in base],
         [b["tempo_h2d_ms"] + b["tempo_d2h_ms"] for b in base],
         marker="^", linestyle=":", color="gray",
         label="Transferências H2D + D2H")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Tamanho do vetor (N)")
plt.ylabel("Tempo (ms)")
plt.title("Soma par/ímpar - tempo de execução\n" + titulo_gpu)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa16_tempos.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 2 - Largura de banda efetiva x N
# ============================================================
plt.figure(figsize=(12, 7))
for nome, pts in series:
    plt.plot([p["n"] for p in pts], [p["banda_gbs"] for p in pts],
             marker="o", linewidth=2, label=nome)
plt.xscale("log")
plt.xlabel("Tamanho do vetor (N)")
plt.ylabel("Largura de banda efetiva (GB/s)  [3·N·4 bytes / tempo]")
plt.title("Soma par/ímpar - largura de banda efetiva\n" + titulo_gpu)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa16_banda.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 3 - Tamanho do bloco x tempo e ocupacao (maior N)
# ============================================================
n_max = ns[-1]
fig, ax1 = plt.subplots(figsize=(11, 6))
for kernel, cor in (("sem_desvio", "tab:blue"), ("divergente", "tab:red")):
    pts = [r for r in res if r["n"] == n_max and r["kernel"] == kernel
           and r["config"] == "uma_thread_por_elemento"]
    pts.sort(key=lambda r: r["bloco"])
    ax1.plot([str(p["bloco"]) for p in pts], [p["tempo_kernel_ms"] for p in pts],
             marker="o", linewidth=2, color=cor,
             label=f"Tempo - {kernel.replace('_', ' ')}")
ax1.set_xlabel("Threads por bloco")
ax1.set_ylabel("Tempo do kernel (ms)")

ax2 = ax1.twinx()
pts = [r for r in res if r["n"] == n_max and r["kernel"] == "sem_desvio"
       and r["config"] == "uma_thread_por_elemento"]
pts.sort(key=lambda r: r["bloco"])
ax2.bar([str(p["bloco"]) for p in pts], [p["ocupacao_teorica_pct"] for p in pts],
        alpha=0.25, color="tab:green", label="Ocupação teórica (%)")
ax2.set_ylabel("Ocupação teórica (%)")
ax2.set_ylim(0, 110)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center")
ax1.grid(True, linestyle="--", alpha=0.5)
plt.title(f"Impacto do tamanho do bloco (N = {n_max:,})\n{titulo_gpu}")
fig.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa16_blocos.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 4 - Divergencia: tempo(divergente) / tempo(sem desvio)
# ============================================================
plt.figure(figsize=(11, 6))
for config, bloco, nome in (
        ("uma_thread_por_elemento", 256, "Bloco 256 (1 thread/elemento)"),
        ("dispositivo_grid_stride", None, "Propriedades do dispositivo")):
    a = {p["n"]: p["tempo_kernel_ms"] for p in serie(config, "divergente", bloco)}
    b = {p["n"]: p["tempo_kernel_ms"] for p in serie(config, "sem_desvio", bloco)}
    comuns = sorted(set(a) & set(b))
    plt.plot(comuns, [a[n] / b[n] for n in comuns], marker="o", linewidth=2,
             label=nome)
plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
plt.xscale("log")
plt.xlabel("Tamanho do vetor (N)")
plt.ylabel("Tempo divergente / tempo sem desvio")
plt.title("Efeito do desvio (if/else par-ímpar) dentro do warp\n" + titulo_gpu)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa16_divergencia.png"), dpi=300)
plt.close()

print("Graficos salvos em", PASTA)
